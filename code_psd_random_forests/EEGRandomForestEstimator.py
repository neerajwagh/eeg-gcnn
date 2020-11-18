from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# disable matplotlib plotting on headless server
import matplotlib as mpl
mpl.use('Agg')

import numpy as np
import pandas as pd

# https://scikit-learn.org/stable/developers/develop.html
class EEGRandomForestEstimator(BaseEstimator, ClassifierMixin):
	
	def __init__(self, hyperparams, seed, experiment_name="",
				X = None, y = None, train_indices = None, test_indices = None, train_subject_indices = None, 
				dataset_index=None, test_subject_indices = None, num_folds=None, fold_idx = None,
				heldout_test_subjects=None, heldout_test_indices=None):

		self.hyperparams = hyperparams
		self.model = RandomForestClassifier(n_estimators=hyperparams["n_estimators"], max_depth=hyperparams["max_depth"], 
									   max_features=hyperparams["max_features"], random_state=seed, 
									   class_weight=hyperparams["class_weight"], n_jobs=-1, verbose=0, oob_score=True,
									  bootstrap=True, max_samples=hyperparams["max_samples"],
									  criterion=hyperparams["criterion"], min_samples_leaf=hyperparams["min_samples_leaf"], 
									  min_samples_split=hyperparams["min_samples_split"], ccp_alpha=hyperparams["ccp_alpha"])
		self.experiment_name = experiment_name

		# these change with each fold and are set by CustomGridSearchCV._fit_and_score(), init with None
		self.X = None
		self.y = None
		self.train_indices = None
		self.test_indices = None
		self.train_subject_indices = None
		self.test_subject_indices = None
		self.dataset_index = None
		# self.label_mapping = None
		self.num_folds = None
		self.fold_idx = None
		self.heldout_test_subjects = None
		self.heldout_test_indices = None

		# metrics filled in by self.fit()
		self.oob_score = None
		self.auroc_train = None
		self.auroc_test = None
		self.auroc_patient_train = None
		self.auroc_patient_test = None

		return

	def fit(self):
		
		print("HYPERPARAMS: ", self.hyperparams)
		
		X_train = self.X[self.train_indices, ...]
		# n_indices x n_sensors x n_freqs x n_timepoints
		# n_indices, n_sensors, n_freqs, n_timepoints = X_train.shape
		# X_train = X_train.reshape(n_indices, (n_sensors*n_freqs*n_timepoints))
		y_train = self.y[self.train_indices]
		

		self.model.fit(X_train, y_train)

		self.oob_score = self.model.oob_score_
		print("OOB Score: ", self.model.oob_score_)
		print("Classes: ", self.model.classes_)

		X_test = self.X[self.test_indices, ...]
		# n_indices, n_sensors, n_freqs, n_timepoints = X_test.shape
		# X_test = X_test.reshape(n_indices, (n_sensors*n_freqs*n_timepoints))
		y_test = self.y[self.test_indices]

		# Confusion-matrix based metrics
		# y_predict = self.model.predict(X_test)
		# print(classification_report(y_test, y_predict))

		# window-level AUROC
		y_probs_test = self.model.predict_proba(X_test)
		# fpr, tpr, _ = roc_curve(y_test, y_score[:, 1], pos_label="healthy")
		# roc_auc = auc(fpr, tpr)
		# self.auroc_test = roc_auc
		# print("window-level validation set AUC: ", roc_auc)

		y_probs_train = self.model.predict_proba(X_train)
		# fpr, tpr, _ = roc_curve(y_train, y_score[:, 1], pos_label="healthy")
		# roc_auc = auc(fpr, tpr)
		# self.auroc_train = roc_auc
		# print("window-level training set AUC: ", roc_auc)

		self.collect_metrics(y_probs_train=y_probs_train,
							y_true_train=y_train, 
							# y_pred_train, 
							sample_indices_train=self.train_indices,
							y_probs_test=y_probs_test, 
							y_true_test=y_test, 
							# y_pred_test, 
							sample_indices_test=self.test_indices,
							fold_idx=self.fold_idx)

		return self
	
	# create patient-level metrics
	def get_patient_prediction(self, df):
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
		return np.array(list(zip(return_df["y_probs_0"], return_df["y_probs_1"]))), list(return_df["y_true"])
		# , list(return_df["y_pred"])

	# record all the metrics on both train and validation sets
	def collect_metrics(self, y_probs_train, y_true_train, 
						# y_pred_train, 
						sample_indices_train,
						y_probs_test, y_true_test, 
						# y_pred_test, 
						sample_indices_test,
						fold_idx):

		# create patient-level train and test dataframes
		rows = [ ]
		for i in range(len(sample_indices_train)):
			idx = sample_indices_train[i]
			temp = { }
			# temp["patient_ID"] = self.dataset_index.iloc[idx, "patient_ID"]
			temp["patient_ID"] = str(self.dataset_index.ix[idx, "patient_ID"])
			temp["sample_idx"] = idx
			temp["y_true"] = y_true_train[i]
			temp["y_probs_0"] = y_probs_train[i, 0]
			temp["y_probs_1"] = y_probs_train[i, 1]
			# temp["y_pred"] = y_pred_train[i]
			rows.append(temp)
		train_patient_df = pd.DataFrame(rows)

		rows = [ ]
		for i in range(len(sample_indices_test)):
			idx = sample_indices_test[i]
			temp = { }
			# temp["patient_ID"] = self.dataset_index.iloc[idx, "patient_ID"]
			temp["patient_ID"] = str(self.dataset_index.ix[idx, "patient_ID"])
			temp["sample_idx"] = idx
			temp["y_true"] = y_true_test[i]
			temp["y_probs_0"] = y_probs_test[i, 0]
			temp["y_probs_1"] = y_probs_test[i, 1]
			# temp["y_pred"] = y_pred_test[i]
			rows.append(temp)
		test_patient_df = pd.DataFrame(rows)

		# get patient-level metrics from window-level dataframes
		y_probs_train_patient, y_true_train_patient = self.get_patient_prediction(train_patient_df)
		y_probs_test_patient, y_true_test_patient = self.get_patient_prediction(test_patient_df)

		# AUROC
		from sklearn.metrics import roc_auc_score
		# CAUTION - The binary case expects a shape (n_samples,), and the scores must be the scores of the class with the greater label.
		# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
		self.auroc_train = roc_auc_score(y_true_train, y_probs_train[:,1])
		self.auroc_test = roc_auc_score(y_true_test, y_probs_test[:,1])

		# PATIENT-LEVEL AUROC
		from sklearn.metrics import roc_auc_score
		self.auroc_patient_train = roc_auc_score(y_true_train_patient, y_probs_train_patient[:,1])
		self.auroc_patient_test = roc_auc_score(y_true_test_patient, y_probs_test_patient[:,1])

		print("[ESTIMATOR:{0}][TRAIN] auroc: {1:1.3f}".format(
			self.fold_idx,
			self.auroc_train
		))
		print("[ESTIMATOR:{0}][VALID] auroc: {1:1.3f}".format(
			self.fold_idx,
			self.auroc_test
		))
		print("[ESTIMATOR:{0}][PATIENT-TRAIN] auroc: {1:1.3f}".format(
			self.fold_idx,
			self.auroc_patient_train
		))
		print("[ESTIMATOR:{0}][PATIENT-VALID] auroc: {1:1.3f}".format(
			self.fold_idx,
			self.auroc_patient_test
		))

		return

	def get_training_history(self):
		print("[ESTIMATOR:{}] returning training history and metrics to CV call...".format(self.fold_idx))
		return ({
		# loss on balanced mini-batches
		"oob_score" : self.oob_score,
		# window-level metrics
		"auroc_train" : self.auroc_train,
		"auroc_test" : self.auroc_test,
		# patient-level metrics
		"auroc_patient_train" : self.auroc_patient_train,
		"auroc_patient_test" : self.auroc_patient_test
		})
