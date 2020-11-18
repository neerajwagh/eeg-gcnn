import os
import numpy as np
import pandas as pd
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, auc, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

if __name__ == "__main__":
	
	# choose between window-level simulation or subject-level simulations
	WINDOW_LEVEL_SIMULATION = False
	# choose which class is considered positive
	IS_POSITIVE_CLASS_MAJORITY = True
	# NOTE: even though the positive class may change when calculating metrics, the label encoding is fixed: 0 - diseased, 1 - healthy
	# pos_label = 0 if positive class is diseased (majority), = 1 when positive class is healthy (minority)
	POS_LABEL = int(not IS_POSITIVE_CLASS_MAJORITY)
	POS_CLASS = "diseased" if POS_LABEL == 0 else "healthy"
	# 2 types of chance models = "predict-imbalance-probability" or "predict-all-as-majority-class"
	# MODEL_TYPE = "predict-imbalance-probability"
	MODEL_TYPE = "predict-all-as-majority-class"

	print("** LABEL ENCODING IS FIXED: HEALTHY = 1, DISEASED = 0 **\n")
	print(f"** METRICS CALCULATION: POSITIVE CLASS = {POS_CLASS}, IS POSITIVE CLASS MAJORITY = {IS_POSITIVE_CLASS_MAJORITY} **\n")
	print(f"** CHOSEN TRIVIAL MODEL: {MODEL_TYPE} **\n")

	MASTER_DATASET_INDEX = pd.read_csv("master_metadata_index.csv")
	subjects = MASTER_DATASET_INDEX["patient_ID"].astype("str").unique()
	print("[MAIN] Subject list fetched! Total subjects are {}...".format(len(subjects)))

	# CAUTION: splitting whole subjects into train+validation and heldout test
	SEED = 42
	train_subjects, test_subjects = train_test_split(subjects, test_size=0.30, random_state=SEED)
	print("[MAIN] (Train + validation) and (heldout test) split made at subject level. 30 percent subjects held out for testing.")

	if WINDOW_LEVEL_SIMULATION:
		print("** WINDOW-LEVEL SIMULATIONS **\n")

		NUM_TEST_SAMPLES = 68778
		# imbalace factor = #diseased/#healthy
		IMBALANCE_FACTOR = 8.96220
		# prob threshold = #minority/(#minority + #majority) - TAKEN FROM TRAINING SET WINDOWS
		# interpretation depends on whether minority or majority is the positive class
		PREDICTION_PROBABILITY_THRESHOLD = 0.100379

		# get indices for test subjects!
		heldout_test_indices = MASTER_DATASET_INDEX.index[MASTER_DATASET_INDEX["patient_ID"].astype("str").isin(test_subjects)].tolist()
		y = load("labels_y", mmap_mode='r')
		label_mapping, y = np.unique(y, return_inverse = True)
		print("[MAIN] unique labels to [0 1] mapping:", label_mapping)
		truth_labels = np.array(y[heldout_test_indices])
	
	else:
		print("** SUBJECT-LEVEL SIMULATIONS **\n")

		# subject-level simulations!
		NUM_TEST_SAMPLES = 478
		# imbalace factor = #diseased/#healthy - TAKEN FROM TRAINING SET SUBJECTS!
		IMBALANCE_FACTOR = 6.384105	
		# prob threshold = #minority/(#minority + #majority) - TAKEN FROM TRAINING SET SUBJECTS
		# interpretation depends on whether minority or majority is the positive class
		PREDICTION_PROBABILITY_THRESHOLD = 0.135426

		# NOTE: labeling healthy = 1, diseased = 0, consistent with the if clause
		truth_labels = np.array([1 if "sub-" in x else 0 for x in test_subjects])

	
	SEED = 42
	np.random.seed(SEED)
	print ("GROUND TRUTH LABELS: ", np.unique(truth_labels, return_counts=True))
	assert len(truth_labels) == NUM_TEST_SAMPLES

	if MODEL_TYPE == "predict-imbalance-probability":

		# run simulations for multiple seeds/multiple times
		precision_scores = [ ]
		recall_scores = [ ]
		f1_scores = [ ]
		bal_acc_scores = [ ]
		auroc_scores = [ ]

		for i in range(1000):

			# make chance-level predictions with a blind model - predict positive class with imbalance probability
			# NOTE: ASSUMING TEST DISTRIBUTION FOLLOWS THE TRAINING LABEL DISTRIBUTION! (which it does for the scope of the paper)
			predictions = np.random.choice([0, 1], NUM_TEST_SAMPLES, p=[(1-PREDICTION_PROBABILITY_THRESHOLD), PREDICTION_PROBABILITY_THRESHOLD])			
			# class probability for the greater label (0) 
			prediction_probabilites = np.array([1.0 if x == 0 else 0.0 for x in list(predictions)])
			print ("PREDICTIONS: ", np.unique(predictions, return_counts=True))
			# print ("PREDICTION PROBA: ", np.unique(prediction_probabilites, return_counts=True))

			# get subject-level metrics
			precision_test =  precision_score(truth_labels, predictions, pos_label=POS_LABEL)
			recall_test =  recall_score(truth_labels, predictions, pos_label=POS_LABEL)
			f1_test = f1_score(truth_labels, predictions, pos_label=POS_LABEL)
			bal_acc_test = balanced_accuracy_score(truth_labels, predictions)
			auroc_test = roc_auc_score(truth_labels, prediction_probabilites)

			precision_scores.append(precision_test)
			recall_scores.append(recall_test)
			f1_scores.append(f1_test)
			bal_acc_scores.append(bal_acc_test)
			auroc_scores.append(auroc_test)
		
		# print mean and std. dev. across all simulations
		import statistics as stats
		print(f"PRECISION: {stats.mean(precision_scores)} ({stats.stdev(precision_scores)})")
		print(f"RECALL: {stats.mean(recall_scores)} ({stats.stdev(recall_scores)})")
		print(f"F-1: {stats.mean(f1_scores)} ({stats.stdev(f1_scores)})")
		print(f"BALANCED ACCURACY: {stats.mean(bal_acc_scores)} ({stats.stdev(bal_acc_scores)})")
		print(f"AUC: {stats.mean(auroc_scores)} ({stats.stdev(auroc_scores)})")
		print("[MAIN] exiting...")

	elif MODEL_TYPE == "predict-all-as-majority-class":
		predictions = np.zeros((NUM_TEST_SAMPLES, ), dtype=int)
		# probabilities for the greater class (0) only, therefore 1.0
		prediction_probabilites = np.ones((NUM_TEST_SAMPLES, ), dtype=float)
		print ("PREDICTIONS: ", np.unique(predictions, return_counts=True))

		# get subject-level metrics
		precision_test =  precision_score(truth_labels, predictions, pos_label=POS_LABEL)
		recall_test =  recall_score(truth_labels, predictions, pos_label=POS_LABEL)
		f1_test = f1_score(truth_labels, predictions, pos_label=POS_LABEL)
		bal_acc_test = balanced_accuracy_score(truth_labels, predictions)
		auroc_test = roc_auc_score(truth_labels, prediction_probabilites)

		print(f"Precision: {precision_test}\nRecall: {recall_test}\nF-1: {f1_test}\nBalanced Accuracy: {bal_acc_test}\nAUC: {auroc_test}")
		print("[MAIN] exiting...")

