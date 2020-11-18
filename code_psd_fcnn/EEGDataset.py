# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

from mne.time_frequency import stft, stftfreq
from torch.utils.data import Dataset
import torch
import numpy as np

class EEGDataset(Dataset):
	def __init__(self, X, y, indices, loader_type, sfreq, transform=None):

		# CAUTION - epochs and labels are memory-mapped, used as if they are in RAM.
		self.epochs = X
		self.labels = y
		self.indices = indices
		self.sfreq = sfreq
		self.loader_type = loader_type
		self.transform = transform
		return None
	
	# return the total samples in the current designated fold
	def __len__(self):
		return len(self.indices)
	
	# retrieve one sample from the dataset after applying all transforms
	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		# map input idx (ranging from 0 to __len__() inside self.indices) to an idx in the whole dataset (inside self.epochs)
		# assert idx < len(self.indices)
		idx = self.indices[idx]

		sample = {
				"psd_features" : np.array(self.epochs[idx, :]), 
				"labels" : np.array(self.labels[idx]),
				"dataset_idx" : idx
				}
		return sample