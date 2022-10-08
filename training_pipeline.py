from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
from joblib import load
import statistics as stats
from sklearn import preprocessing

import torch.backends.cudnn as cudnn
cudnn.enabled = True
cudnn.benchmark = False
cudnn.deterministic = True

from code_psd_shallow_eeg_gcnn.EEGGraphDataset import EEGGraphDataset
from code_psd_shallow_eeg_gcnn.EEGGraphConvNet import EEGGraphConvNet
from torch_geometric.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import make_scorer
from sklearn.metrics import balanced_accuracy_score, auc, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from torchvision.transforms import Compose, ToTensor

stats_test_data = { }

# after each epoch, record all the metrics on both train and validation sets
def collect_metrics(y_probs_test, y_true_test, y_pred_test, sample_indices_test,
					fold_idx, experiment_name):

	dataset_index = pd.read_csv("master_metadata_index.csv", dtype={"patient_ID":str, })

	# create patient-level train and test dataframes
	rows = [ ]
	for i in range(len(sample_indices_test)):
		idx = sample_indices_test[i]
		temp = { }
		temp["patient_ID"] = str(dataset_index.loc[idx, "patient_ID"])
		temp["sample_idx"] = idx
		temp["y_true"] = y_true_test[i]
		temp["y_probs_0"] = y_probs_test[i, 0]
		temp["y_probs_1"] = y_probs_test[i, 1]
		temp["y_pred"] = y_pred_test[i]
		rows.append(temp)
	test_patient_df = pd.DataFrame(rows)

	# get patient-level metrics from window-level dataframes
	y_probs_test_patient, y_true_test_patient, y_pred_test_patient = get_patient_prediction(test_patient_df, fold_idx)

	stats_test_data[f"probs_0_fold_{fold_idx}"] = y_probs_test_patient[:, 0]
	stats_test_data[f"probs_1_fold_{fold_idx}"] = y_probs_test_patient[:, 1]

	window_csv_dict = { }
	patient_csv_dict = { }

	# WINDOW-LEVEL ROC PLOT
	# pos_label="healthy"
	fpr, tpr, thresholds = roc_curve(y_true_test, y_probs_test[:,1], pos_label=1)
	window_csv_dict[f"fpr_fold_{fold_idx}"] = fpr
	window_csv_dict[f"tpr_fold_{fold_idx}"] = tpr
	window_csv_dict[f"thres_fold_{fold_idx}"] = thresholds

	# PATIENT-LEVEL ROC PLOT - select optimal threshold for this, and get patient-level precision, recall, f1
	# pos_label="healthy"
	fpr, tpr, thresholds = roc_curve(y_true_test_patient, y_probs_test_patient[:,1], pos_label=1)
	patient_csv_dict[f"fpr_fold_{fold_idx}"] = fpr
	patient_csv_dict[f"tpr_fold_{fold_idx}"] = tpr
	patient_csv_dict[f"thres_fold_{fold_idx}"] = thresholds

	# select an optimal threshold using the ROC curve
	# Youden's J statistic to obtain the optimal probability threshold and this method gives equal weights to both false positives and false negatives
	optimal_proba_cutoff = sorted(list(zip(np.abs(tpr - fpr), thresholds)), key=lambda i: i[0], reverse=True)[0][1]
	# print (optimal_proba_cutoff)

	# calculate class predictions and confusion-based metrics using the optimal threshold
	roc_predictions = [1 if i >= optimal_proba_cutoff else 0 for i in y_probs_test_patient[:,1]]

	precision_patient_test =  precision_score(y_true_test_patient, roc_predictions, pos_label=0)
	recall_patient_test =  recall_score(y_true_test_patient, roc_predictions, pos_label=0)
	f1_patient_test = f1_score(y_true_test_patient, roc_predictions, pos_label=0)
	bal_acc_patient_test = balanced_accuracy_score(y_true_test_patient, roc_predictions)


	# PATIENT-LEVEL AUROC
	from sklearn.metrics import roc_auc_score
	auroc_patient_test = roc_auc_score(y_true_test_patient, y_probs_test_patient[:,1])

	# AUROC
	from sklearn.metrics import roc_auc_score
	# CAUTION - The binary case expects a shape (n_samples,), and the scores must be the scores of the class with the greater label.
	# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
	auroc_test = roc_auc_score(y_true_test, y_probs_test[:,1])
	
	return auroc_patient_test, auroc_test, precision_patient_test, recall_patient_test, f1_patient_test, bal_acc_patient_test

# create patient-level metrics
def get_patient_prediction(df, fold_idx):
	unique_patients = list(df["patient_ID"].unique())
	grouped_df = df.groupby("patient_ID")
	rows = [ ]
	for patient in unique_patients:
		patient_df = grouped_df.get_group(patient)
		temp = { }
		temp["patient_ID"] = patient
		temp["y_true"] = list(patient_df["y_true"].unique())[0]
		assert len(list(patient_df["y_true"].unique())) == 1
		temp["y_pred"] = patient_df["y_pred"].mode()[0]
		temp["y_probs_0"] = patient_df["y_probs_0"].mean()
		temp["y_probs_1"] = patient_df["y_probs_1"].mean()
		rows.append(temp)
	return_df = pd.DataFrame(rows)

	# need subject names and labels for comparisons testing
	if fold_idx == 0:
		stats_test_data["subject_id"] = list(return_df["patient_ID"][:])
		stats_test_data["label"] = return_df["y_true"][:]

	return np.array(list(zip(return_df["y_probs_0"], return_df["y_probs_1"]))), list(return_df["y_true"]), list(return_df["y_pred"])


if __name__ == "__main__":

	GPU_IDX = 0
	EXPERIMENT_NAME = "psd_gnn_shallow"
	BATCH_SIZE = 512
	SFREQ = 250.0
	NUM_EPOCHS = 100
	NUM_WORKERS = 6
	PIN_MEMORY = True

	# ensure reproducibility of results
	SEED = 42
	np.random.seed(SEED)
	torch.manual_seed(SEED)
	print("[MAIN] Numpy and PyTorch seed set to {} for reproducibility.".format(SEED))

	MASTER_DATASET_INDEX = pd.read_csv("master_metadata_index.csv", dtype={"patient_ID":str, })
	subjects = MASTER_DATASET_INDEX["patient_ID"].astype("str").unique()
	print("[MAIN] Subject list fetched! Total subjects are {}...".format(len(subjects)))

	# NOTE: splitting whole subjects into train+validation and heldout test
	train_val_subjects, test_subjects = train_test_split(subjects, test_size=0.30, random_state=SEED)
	print("[MAIN] (Train + validation) and (heldout test) split made at subject level. 30 percent subjects held out for testing.")	
	train_subjects, val_subjects = train_test_split(train_val_subjects, test_size=0.20, random_state=SEED)
	train_indices = MASTER_DATASET_INDEX.index[MASTER_DATASET_INDEX["patient_ID"].astype("str").isin(train_subjects)].tolist()
	val_indices = MASTER_DATASET_INDEX.index[MASTER_DATASET_INDEX["patient_ID"].astype("str").isin(val_subjects)].tolist()

	# use GPU when available
	DEVICE = torch.device('cuda:{}'.format(GPU_IDX) if torch.cuda.is_available() else 'cpu')
	torch.cuda.set_device(DEVICE)
	print('[MAIN] Using device:', DEVICE, torch.cuda.get_device_name(DEVICE))
	
	X = load("psd_features_data_X")
	y = load("labels_y")

	# normalize psd_features_data_X
	normd_x = []
	for i in range(len(y)):
		arr = X[i, :]
		arr = arr.reshape(1, -1)
		arr2 = preprocessing.normalize(arr)
		arr2 = arr2.reshape(48)
		normd_x.append(arr2)
	
	norm = np.array(normd_x)
	X = norm.reshape(len(y), 48)

	# get 0/1 labels for pytorch, ensure mapping is the same between train and test
	label_mapping, y = np.unique(y, return_inverse = True)
	print("[MAIN] unique labels to [0 1] mapping:", label_mapping)

	model = EEGGraphConvNet(reduced_sensors=False)
	model = model.to(DEVICE).double()

	labels_unique, counts = np.unique(y, return_counts=True)

	class_weights = np.array([1.0/x for x in counts])
	# provide weights for samples in the training set only		
	sample_weights = class_weights[y[train_indices]]
	# sampler needs to come up with training set size number of samples
	weighted_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_indices), replacement=True)

	# define training set
	train_dataset = EEGGraphDataset(X=X, y=y, indices=train_indices, loader_type="train", 
									sfreq=SFREQ, transform=Compose([ToTensor()]))
	train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, sampler=weighted_sampler,
							 num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
	
	# define validation set
	val_dataset = EEGGraphDataset(X=X, y=y, indices=val_indices, loader_type="validation", 
									sfreq=SFREQ, transform=Compose([ToTensor()]))
	val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, 
							  shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

	# define loss function
	loss_function = torch.nn.CrossEntropyLoss()
	# define optimizer
	optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
	# define scheduler
	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[i*10 for i in range(1, 26)], gamma=0.1)

	# start training
	for epoch in range(NUM_EPOCHS):

		model.train()
		train_loss = []
		val_loss = []

		y_probs_train = torch.empty(0, 2).to(DEVICE)

		y_true_train = [ ]
		y_pred_train = [ ]
		window_indices_train = [ ]

		for batch_idx, batch in enumerate(train_loader):

			# send batch to GPU
			X_batch = batch.to(device=DEVICE, non_blocking=True)
			y_batch = torch.tensor(batch.y)
			y_batch = y_batch.to(device=DEVICE, non_blocking=True)
			window_indices_train += X_batch.dataset_idx.cpu().numpy().tolist()
			optimizer.zero_grad()

			# forward pass
			outputs = model(X_batch.x, X_batch.edge_index, X_batch.edge_attr, X_batch.batch).float()
			loss = loss_function(outputs, y_batch)
			train_loss.append(loss.item())
			# backward pass
			loss.backward()

			_, predicted = torch.max(outputs.data, 1)
			y_pred_train += predicted.cpu().numpy().tolist()

			# concatenate along 0th dimension
			y_probs_train = torch.cat((y_probs_train, outputs.data), 0)
			y_true_train += y_batch.cpu().numpy().tolist()

			optimizer.step()
		scheduler.step()

		# returning prob distribution over target classes, take softmax across the 1st dimension
		y_probs_train = torch.nn.functional.softmax(y_probs_train, dim=1).cpu().numpy()
		y_true_train = np.array(y_true_train)

		# calculate training set metrics
		auroc_patient_train, auroc_train, precision_patient_train, recall_patient_train, f1_patient_train, bal_acc_patient_train = collect_metrics(y_probs_test=y_probs_train,
						y_true_test=y_true_train,
						y_pred_test=y_pred_train,
						sample_indices_test = window_indices_train,					
						fold_idx=0,
						experiment_name=EXPERIMENT_NAME)
		
		# evaluate on validation set
		model.eval()
		with torch.no_grad():
			y_probs_val = torch.empty(0, 2).to(DEVICE)

			y_true_val = [ ]
			y_pred_val = [ ]
			window_indices_val = [ ]

			for i, batch in enumerate(val_loader):
				X_batch = batch.to(device=DEVICE, non_blocking=True)
				y_batch = torch.tensor(batch.y)
				y_batch = y_batch.to(device=DEVICE, non_blocking=True)
				window_indices_val += X_batch.dataset_idx.cpu().numpy().tolist()
				outputs = model(X_batch.x, X_batch.edge_index, X_batch.edge_attr, X_batch.batch).float()

				loss = loss_function(outputs, y_batch)
				val_loss.append(loss.item())

				_, predicted = torch.max(outputs.data, 1)
				y_pred_val += predicted.cpu().numpy().tolist()

				# concatenate along 0th dimension
				y_probs_val = torch.cat((y_probs_val, outputs.data), 0)
				y_true_val += y_batch.cpu().numpy().tolist()

		# returning prob distribution over target classes, take softmax across the 1st dimension
		y_probs_val = torch.nn.functional.softmax(y_probs_val, dim=1).cpu().numpy()
		y_true_val = np.array(y_true_val)

		# get validation set metrics
		auroc_patient_val, auroc_val, precision_patient_val, recall_patient_val, f1_patient_val, bal_acc_patient_val = collect_metrics(y_probs_test=y_probs_val,
						y_true_test=y_true_val,
						y_pred_test=y_pred_val,
						sample_indices_test = val_indices,					
						fold_idx=0,
						experiment_name=EXPERIMENT_NAME)
		
		# save the model every 20 epochs
		if epoch % 20 == 0:
			state = {
				'model_description': str(model),
				'state_dict': model.state_dict(),
				'optimizer': optimizer.state_dict()
			}

			torch.save(state, f"model_{epoch}.ckpt")

		print(f'Epoch: {epoch}-----------------------------------------------------------')
		print(f"Train loss: {np.mean(train_loss):.3f}; Validation loss: {np.mean(val_loss):.3f}")
		print(f"Train AUROC:{auroc_train:.3f}; Validation AUROC: {auroc_val:.3f}")
		print(f"Train patient metrics: AUROC{auroc_patient_train:.3f}, precision: {precision_patient_train:.3f}, recall: {recall_patient_train:.3f}, f1: {f1_patient_train:.3f}, bal acc: {bal_acc_patient_train:.3f}")
		print(f"Validation patient metrics: AUROC{auroc_patient_val:.3f}, precision: {precision_patient_val:.3f}, recall: {recall_patient_val:.3f}, f1: {f1_patient_val:.3f}, bal acc: {bal_acc_patient_val:.3f}")
