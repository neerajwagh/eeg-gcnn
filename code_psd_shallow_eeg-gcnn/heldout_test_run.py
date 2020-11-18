
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
from joblib import load
import statistics as stats

import torch.backends.cudnn as cudnn
cudnn.enabled = True
cudnn.benchmark = False
cudnn.deterministic = True

from EEGGraphDataset import EEGGraphDataset
from EEGGraphConvNet import EEGGraphConvNet
from torch_geometric.data import DataLoader

from sklearn.metrics import make_scorer
from sklearn.metrics import balanced_accuracy_score, auc, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from torchvision.transforms import Compose, ToTensor

# after each epoch, record all the metrics on both train and validation sets
def collect_metrics(y_probs_test, y_true_test, y_pred_test, sample_indices_test,
					fold_idx, experiment_name):

	dataset_index = pd.read_csv("master_metadata_index.csv")

	# create patient-level train and test dataframes
	rows = [ ]
	for i in range(len(sample_indices_test)):
		idx = sample_indices_test[i]
		temp = { }
		# temp["patient_ID"] = dataset_index.iloc[idx, "patient_ID"]
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
	# fpr, tpr, thresholds = roc_curve(y_true_test, y_probs_test[:,0], pos_label=0)
	window_csv_dict[f"fpr_fold_{fold_idx}"] = fpr
	window_csv_dict[f"tpr_fold_{fold_idx}"] = tpr
	window_csv_dict[f"thres_fold_{fold_idx}"] = thresholds

	# PATIENT-LEVEL ROC PLOT - select optimal threshold for this, and get patient-level precision, recall, f1
	# pos_label="healthy"
	fpr, tpr, thresholds = roc_curve(y_true_test_patient, y_probs_test_patient[:,1], pos_label=1)
	# fpr, tpr, thresholds = roc_curve(y_true_test_patient, y_probs_test_patient[:,0], pos_label=0)
	patient_csv_dict[f"fpr_fold_{fold_idx}"] = fpr
	patient_csv_dict[f"tpr_fold_{fold_idx}"] = tpr
	patient_csv_dict[f"thres_fold_{fold_idx}"] = thresholds

	# select an optimal threshold using the ROC curve
	# Youden's J statistic to obtain the optimal probability threshold and this method gives equal weights to both false positives and false negatives
	optimal_proba_cutoff = sorted(list(zip(np.abs(tpr - fpr), thresholds)), key=lambda i: i[0], reverse=True)[0][1]
	print (optimal_proba_cutoff)

	# calculate class predictions and confusion-based metrics using the optimal threshold
	roc_predictions = [1 if i >= optimal_proba_cutoff else 0 for i in y_probs_test_patient[:,1]]
	# roc_predictions = [0 if i >= optimal_proba_cutoff else 1 for i in y_probs_test_patient[:,0]]

	# precision_patient_test =  precision_score(y_true_test_patient, roc_predictions)
	# recall_patient_test =  recall_score(y_true_test_patient, roc_predictions)
	# f1_patient_test = f1_score(y_true_test_patient, roc_predictions)
	precision_patient_test =  precision_score(y_true_test_patient, roc_predictions, pos_label=0)
	recall_patient_test =  recall_score(y_true_test_patient, roc_predictions, pos_label=0)
	f1_patient_test = f1_score(y_true_test_patient, roc_predictions, pos_label=0)
	bal_acc_patient_test = balanced_accuracy_score(y_true_test_patient, roc_predictions)


	# PATIENT-LEVEL AUROC
	from sklearn.metrics import roc_auc_score
	auroc_patient_test = roc_auc_score(y_true_test_patient, y_probs_test_patient[:,1])
	# auroc_patient_test = roc_auc_score(y_true_test_patient, y_probs_test_patient[:,0])

	# AUROC
	from sklearn.metrics import roc_auc_score
	# CAUTION - The binary case expects a shape (n_samples,), and the scores must be the scores of the class with the greater label.
	# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
	auroc_test = roc_auc_score(y_true_test, y_probs_test[:,1])
	# auroc_test = roc_auc_score(y_true_test, y_probs_test[:,0])	
	print(auroc_patient_test, auroc_test)
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


stats_test_data = { }

if __name__ == "__main__":

	NUM_FOLDS = 10
	GPU_IDX = 0
	EXPERIMENT_NAME = "psd_gnn_shallow"
	BATCH_SIZE = 512
	SFREQ = 250.0
	REDUCED_SENSORS = True

	# ensure reproducibility of results
	SEED = 42
	np.random.seed(SEED)
	torch.manual_seed(SEED)
	print("[MAIN] Numpy and PyTorch seed set to {} for reproducibility.".format(SEED))

	MASTER_DATASET_INDEX = pd.read_csv("master_metadata_index.csv")
	subjects = MASTER_DATASET_INDEX["patient_ID"].astype("str").unique()
	print("[MAIN] Subject list fetched! Total subjects are {}...".format(len(subjects)))

	# NOTE: splitting whole subjects into train+validation and heldout test
	train_subjects, test_subjects = train_test_split(subjects, test_size=0.30, random_state=SEED)
	print("[MAIN] (Train + validation) and (heldout test) split made at subject level. 30 percent subjects held out for testing.")	
	# get indices for test subjects!
	heldout_test_indices = MASTER_DATASET_INDEX.index[MASTER_DATASET_INDEX["patient_ID"].astype("str").isin(test_subjects)].tolist()

	# use GPU when available
	DEVICE = torch.device('cuda:{}'.format(GPU_IDX) if torch.cuda.is_available() else 'cpu')
	torch.cuda.set_device(DEVICE)
	print('[MAIN] Using device:', DEVICE, torch.cuda.get_device_name(DEVICE))
	
	X = load("psd_features_data_X", mmap_mode='r')
	y = load("labels_y", mmap_mode='r')

	# get 0/1 labels for pytorch, ensure mapping is the same between train and test
	label_mapping, y = np.unique(y, return_inverse = True)
	print("[MAIN] unique labels to [0 1] mapping:", label_mapping)

	auroc_patient_test_folds = [ ]
	auroc_test_folds = [ ]
	precision_patient_test_folds = [ ]
	recall_patient_test_folds = [ ]
	f1_patient_test_folds = [ ]
	bal_acc_patient_test_folds = [ ]

	for FOLD_IDX in range(10):

		model = EEGGraphConvNet(reduced_sensors=REDUCED_SENSORS)
		checkpoint = torch.load("./{}_fold_{}.ckpt".format(EXPERIMENT_NAME, FOLD_IDX), map_location=DEVICE)
		model.load_state_dict(checkpoint['state_dict'])
		model = model.to(DEVICE).double()

		NUM_WORKERS = 6
		PIN_MEMORY = True
		heldout_test_dataset = EEGGraphDataset(X=X, y=y, indices=heldout_test_indices, loader_type="heldout_test", 
										sfreq=SFREQ, transform=Compose([ToTensor()]))
		heldout_test_loader = DataLoader(dataset=heldout_test_dataset, batch_size=BATCH_SIZE, 
									shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

		model.eval()
		with torch.no_grad():
			y_probs = torch.empty(0, 2).to(DEVICE)

			y_true = [ ]
			y_pred = [ ]
			window_indices = [ ]

			for i, batch in enumerate(heldout_test_loader):
				batch = batch.to(DEVICE, non_blocking=True)
				X_batch = batch
				y_batch = torch.tensor(batch.y)
				window_indices += batch.dataset_idx.cpu().numpy().tolist()
				outputs = model(X_batch.x, X_batch.edge_index, X_batch.edge_attr, X_batch.batch).float()

				_, predicted = torch.max(outputs.data, 1)
				y_pred += predicted.cpu().numpy().tolist()

				# concatenate along 0th dimension
				y_probs = torch.cat((y_probs, outputs.data), 0)
				y_true += y_batch.cpu().numpy().tolist()

		# returning prob distribution over target classes, take softmax across the 1st dimension
		y_probs = torch.nn.functional.softmax(y_probs, dim=1).cpu().numpy()
		y_true = np.array(y_true)

		auroc_patient_test, auroc_test, precision_patient_test, recall_patient_test, f1_patient_test, bal_acc_patient_test = collect_metrics(y_probs_test=y_probs,
						y_true_test=y_true,
						y_pred_test=y_pred,
						sample_indices_test = heldout_test_indices,					
						fold_idx=FOLD_IDX,
						experiment_name=EXPERIMENT_NAME)

		auroc_patient_test_folds.append(auroc_patient_test)
		auroc_test_folds.append(auroc_test)

		precision_patient_test_folds.append(precision_patient_test)
		recall_patient_test_folds.append(recall_patient_test)
		f1_patient_test_folds.append(f1_patient_test)
		bal_acc_patient_test_folds.append(bal_acc_patient_test)

	print(f"10-folds-avg heldout test AUROC: {stats.mean(auroc_test_folds)} ({stats.stdev(auroc_test_folds)})")
	print(f"10-folds-avg heldout test patient AUROC: {stats.mean(auroc_patient_test_folds)} ({stats.stdev(auroc_patient_test_folds)})")
	print(f"10-folds-avg heldout test patient PRECISION: {stats.mean(precision_patient_test_folds)} ({stats.stdev(precision_patient_test_folds)})")
	print(f"10-folds-avg heldout test patient RECALL: {stats.mean(recall_patient_test_folds)} ({stats.stdev(recall_patient_test_folds)})")
	print(f"10-folds-avg heldout test patient F-1: {stats.mean(f1_patient_test_folds)} ({stats.stdev(f1_patient_test_folds)})")
	print(f"10-folds-avg heldout test patient BALANCED ACCURACY: {stats.mean(bal_acc_patient_test_folds)} ({stats.stdev(bal_acc_patient_test_folds)})")

	print("[MAIN] exiting...")