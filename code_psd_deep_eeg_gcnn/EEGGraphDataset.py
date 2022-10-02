# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data, Dataset

class EEGGraphDataset(Dataset):
	def __init__(self, X, y, indices, loader_type, sfreq, transform=None):

		# CAUTION - epochs and labels are memory-mapped, used as if they are in RAM.
		self.epochs = X
		self.labels = y
		self.indices = indices
		self.sfreq = sfreq
		self.loader_type = loader_type
		self.transform = transform

		# num_sensors = num_nodes = N = 8
		# NOTE: this order decides the node index, keep consistent!
		self.ch_names = [
			"F7-F3", 
			"F8-F4",
			"T7-C3", 
			"T8-C4",
			"P7-P3", 
			"P8-P4",
			"O1-P3", 
			"O2-P4"
		]

		# in the 10-10 system, in between the 2 10-20 electrodes in ch_names, used for calculating edge weights
		self.ref_names = [
			"F5",
			"F6",
			"C5",
			"C6",
			"P5",
			"P6",
			# CAUTION: mapping has exceptional case
			# "PO3", # not available in standard_1010.csv, using O1
			# "PO4"  # not available in standard_1010.csv, using O2
			"O1",
			"O2"
		]

		# edge indices source to target - 2 x E = 2 x 64
		# fully connected undirected graph so 8*8=64 edges
		self.node_ids = range(len(self.ch_names))
		from itertools import product
		self.edge_index = torch.tensor([[a,b] for a, b in product(self.node_ids, self.node_ids)], dtype=torch.long).t().contiguous()
		
		# edge attributes - E x 1
		# only the spatial distance between electrodes for now - standardize between 0 and 1
		self.distances = self.get_sensor_distances()
		a = np.array(self.distances)
		self.distances = (a - np.min(a))/(np.max(a) - np.min(a))
		# self.edge_weights = torch.tensor(self.distances, dtype=torch.long)
		self.spec_coh_values = np.load("spec_coh_values.npy", allow_pickle=True)

	# sensor distances don't depend on window ID
	def get_sensor_distances(self):
		import pandas as pd
		coords_1010 = pd.read_csv("standard_1010.tsv.txt", sep='\t')
		num_edges = self.edge_index.shape[1]
		distances = [ ]
		for edge_idx in range(num_edges): 
			sensor1_idx = self.edge_index[0, edge_idx]
			sensor2_idx = self.edge_index[1, edge_idx]
			dist = self.get_geodesic_distance(sensor1_idx, sensor2_idx, coords_1010)
			distances.append(dist)
		
		assert len(distances) == num_edges
		return distances

	def get_geodesic_distance(self, montage_sensor1_idx, montage_sensor2_idx, coords_1010):
		
		# get the reference sensor in the 10-10 system for the current montage pair in 10-20 system
		ref_sensor1 = self.ref_names[montage_sensor1_idx]
		ref_sensor2 = self.ref_names[montage_sensor2_idx]
		
		x1 = float(coords_1010[coords_1010.label == ref_sensor1]["x"])
		y1 = float(coords_1010[coords_1010.label == ref_sensor1]["y"])
		z1 = float(coords_1010[coords_1010.label == ref_sensor1]["z"])

		# print(ref_sensor2, montage_sensor2_idx, coords_1010[coords_1010.label == ref_sensor2]["x"])
		x2 = float(coords_1010[coords_1010.label == ref_sensor2]["x"])
		y2 = float(coords_1010[coords_1010.label == ref_sensor2]["y"])
		z2 = float(coords_1010[coords_1010.label == ref_sensor2]["z"])

		# https://math.stackexchange.com/questions/1304169/distance-between-two-points-on-a-sphere
		import math
		r = 1 # since coords are on unit sphere
		# rounding is for numerical stability, domain is [-1, 1]		
		dist = r * math.acos(round(((x1 * x2) + (y1 * y2) + (z1 * z2)) / (r**2), 2))
		return dist

	# returns size of dataset = number of epochs
	def __len__(self):
		return len(self.indices)

	# retrieve one sample from the dataset after applying all transforms
	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		# map input idx (ranging from 0 to __len__() inside self.indices) to an idx in the whole dataset (inside self.epochs)
		# assert idx < len(self.indices)
		idx = self.indices[idx]

		node_features = self.epochs[idx]
		node_features = torch.from_numpy(node_features.reshape(8, 6))
		
		# spectral coherence between 2 montage channels!
		spec_coh_values = self.spec_coh_values[idx, :]
		# combine edge weights and spect coh values into one value/ one E x 1 tensor
		edge_weights = self.distances + spec_coh_values
		edge_weights = torch.tensor(edge_weights)

		# NOTE: taken from https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#
		# https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html
		data = Data(x=node_features, 
					edge_index=self.edge_index, 
					edge_attr=edge_weights,
					dataset_idx=idx, 
					y=self.labels[idx]
					# pos=None, norm=None, face=None, **kwargs
					)
		return data