from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from joblib import load

from EEGRandomForestEstimator import EEGRandomForestEstimator

from sklearn.metrics import make_scorer
from sklearn.metrics import balanced_accuracy_score, auc, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

def collect_metrics(y_probs_test, y_true_test, 
					# y_pred_test, 
					sample_indices_test,
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
		# temp["y_pred"] = y_pred_test[i]
		rows.append(temp)
	test_patient_df = pd.DataFrame(rows)

	# get patient-level metrics from window-level dataframes
	y_probs_test_patient, y_true_test_patient = get_patient_prediction(test_patient_df, fold_idx)

	# add y_probs_test_patient to csv for statistical testing
	stats_test_data[f"probs_0_fold_{fold_idx}"] = y_probs_test_patient[:, 0]
	stats_test_data[f"probs_1_fold_{fold_idx}"] = y_probs_test_patient[:, 1]

	window_csv_dict = { }
	patient_csv_dict = { }

	# WINDOW-LEVEL ROC PLOT
	# pos_label="healthy"
	fpr, tpr, thresholds = roc_curve(y_true_test, y_probs_test[:,1], pos_label="healthy")
	window_csv_dict[f"fpr_fold_{fold_idx}"] = fpr
	window_csv_dict[f"tpr_fold_{fold_idx}"] = tpr
	window_csv_dict[f"thres_fold_{fold_idx}"] = thresholds

	# PATIENT-LEVEL ROC PLOT
	fpr, tpr, thresholds = roc_curve(y_true_test_patient, y_probs_test_patient[:,1], pos_label="healthy")
	patient_csv_dict[f"fpr_fold_{fold_idx}"] = fpr
	patient_csv_dict[f"tpr_fold_{fold_idx}"] = tpr
	patient_csv_dict[f"thres_fold_{fold_idx}"] = thresholds


	# select an optimal threshold using the ROC curve
	# Youden's J statistic to obtain the optimal probability threshold and this method gives equal weights to both false positives and false negatives
	optimal_proba_cutoff = sorted(list(zip(np.abs(tpr - fpr), thresholds)), key=lambda i: i[0], reverse=True)[0][1]
	print (optimal_proba_cutoff)

	# calculate class predictions and confusion-based metrics using the optimal threshold
	# roc_predictions = [1 if i >= optimal_proba_cutoff else 0 for i in y_probs_test_patient[:,1]]
	roc_predictions = ['healthy' if i >= optimal_proba_cutoff else 'diseased' for i in y_probs_test_patient[:,1]]

	# precision_patient_test =  precision_score(y_true_test_patient, roc_predictions, pos_label="healthy")
	# recall_patient_test =  recall_score(y_true_test_patient, roc_predictions, pos_label="healthy")
	# f1_patient_test = f1_score(y_true_test_patient, roc_predictions, pos_label="healthy")

	precision_patient_test =  precision_score(y_true_test_patient, roc_predictions, pos_label="diseased")
	recall_patient_test =  recall_score(y_true_test_patient, roc_predictions, pos_label="diseased")
	f1_patient_test = f1_score(y_true_test_patient, roc_predictions, pos_label="diseased")
	bal_acc_patient_test = balanced_accuracy_score(y_true_test_patient, roc_predictions)


	# PATIENT-LEVEL AUROC
	from sklearn.metrics import roc_auc_score
	auroc_patient_test = roc_auc_score(y_true_test_patient, y_probs_test_patient[:,1])

	# AUROC
	from sklearn.metrics import roc_auc_score
	# CAUTION - The binary case expects a shape (n_samples,), and the scores must be the scores of the class with the greater label.
	# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
	auroc_test = roc_auc_score(y_true_test, y_probs_test[:,1])
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
		# temp["y_pred"] = patient_df["y_pred"].mode()[0]
		temp["y_probs_0"] = patient_df["y_probs_0"].mean()
		temp["y_probs_1"] = patient_df["y_probs_1"].mean()
		rows.append(temp)
	return_df = pd.DataFrame(rows)

	# need subject names and labels for comparisons testing
	if fold_idx == 0:
		stats_test_data["subject_id"] = list(return_df["patient_ID"][:])
		stats_test_data["label"] = return_df["y_true"][:]

	return np.array(list(zip(return_df["y_probs_0"], return_df["y_probs_1"]))), list(return_df["y_true"]) #, list(return_df["y_pred"])

stats_test_data =  { }

if __name__ == "__main__":

	EXPERIMENT_NAME = "psd_rf_tuned"
	NUM_FOLDS = 10

	# ensure reproducibility of results
	SEED = 42
	np.random.seed(SEED)
	print("[MAIN] Numpy and PyTorch seed set to {} for reproducibility.".format(SEED))

	MASTER_DATASET_INDEX = pd.read_csv("master_metadata_index.csv")
	subjects = MASTER_DATASET_INDEX["patient_ID"].astype("str").unique()
	print("[MAIN] Subject list fetched! Total subjects are {}...".format(len(subjects)))

	# NOTE: splitting whole subjects into train+validation and heldout test
	train_subjects, test_subjects = train_test_split(subjects, test_size=0.30, random_state=SEED)
	print("[MAIN] (Train + validation) and (heldout test) split made at subject level. 30 percent subjects held out for testing.")	
	# get indices for test subjects!
	heldout_test_indices = MASTER_DATASET_INDEX.index[MASTER_DATASET_INDEX["patient_ID"].astype("str").isin(test_subjects)].tolist()

	X = load("psd_features_data_X", mmap_mode='r')
	y = load("labels_y", mmap_mode='r')

	auroc_patient_test_folds = [ ]
	auroc_test_folds = [ ]
	precision_patient_test_folds = [ ]
	recall_patient_test_folds = [ ]
	f1_patient_test_folds = [ ]
	bal_acc_patient_test_folds = [ ]


	for FOLD_IDX in range(10):

		estimator = load(f'{EXPERIMENT_NAME}_fold_{FOLD_IDX}.estimator')
		
		y_true = y[heldout_test_indices]
		y_probs = estimator.model.predict_proba(X[heldout_test_indices, :])

		auroc_patient_test, auroc_test, precision_patient_test, recall_patient_test, f1_patient_test, bal_acc_patient_test = collect_metrics(y_probs_test=y_probs,
						y_true_test=y_true,
						# y_pred_test=y_pred,
						sample_indices_test = heldout_test_indices,					
						fold_idx=FOLD_IDX,
						experiment_name=EXPERIMENT_NAME)

		auroc_patient_test_folds.append(auroc_patient_test)
		auroc_test_folds.append(auroc_test)

		precision_patient_test_folds.append(precision_patient_test)
		recall_patient_test_folds.append(recall_patient_test)
		f1_patient_test_folds.append(f1_patient_test)
		bal_acc_patient_test_folds.append(bal_acc_patient_test)


	import statistics as stats
	print(f"10-folds-avg heldout test patient AUROC: {stats.mean(auroc_patient_test_folds)} ({stats.stdev(auroc_patient_test_folds)})")
	print(f"10-folds-avg heldout test patient PRECISION: {stats.mean(precision_patient_test_folds)} ({stats.stdev(precision_patient_test_folds)})")
	print(f"10-folds-avg heldout test patient RECALL: {stats.mean(recall_patient_test_folds)} ({stats.stdev(recall_patient_test_folds)})")
	print(f"10-folds-avg heldout test patient F-1: {stats.mean(f1_patient_test_folds)} ({stats.stdev(f1_patient_test_folds)})")
	print(f"10-folds-avg heldout test patient BALANCED ACCURACY: {stats.mean(bal_acc_patient_test_folds)} ({stats.stdev(bal_acc_patient_test_folds)})")


	print("[MAIN] exiting...")