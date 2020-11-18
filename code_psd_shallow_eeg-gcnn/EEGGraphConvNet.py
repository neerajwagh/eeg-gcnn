import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm

class EEGGraphConvNet(nn.Module):
	def __init__(self, reduced_sensors, sfreq=None, batch_size=32):
		super(EEGGraphConvNet, self).__init__()
		
		# need these for train_model_and_visualize() function
		self.sfreq = sfreq
		self.batch_size = batch_size
		self.input_size = 8 if reduced_sensors else 62

		self.conv1 = GCNConv(6, 64, improved=True, cached=True, normalize=False)
		self.conv2 = GCNConv(64, 128, improved=True, cached=True, normalize=False)
		self.batchnorm1 = BatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.batchnorm2 = BatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.fc_block1 = nn.Linear(128, 2)

		# Xavier initializations
		self.fc_block1.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if type(x) == nn.Linear else None)

	# NOTE: adapted from https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#learning-methods-on-graphs
	def forward(self, x, edge_index, edge_weight, batch, return_graph_embedding=False):

		x = F.dropout(F.leaky_relu(self.batchnorm1(self.conv1(x, edge_index, edge_weight)), negative_slope=0.01), p=0.5, training=self.training)
		x = F.dropout(F.leaky_relu(self.batchnorm2(self.conv2(x, edge_index, edge_weight)), negative_slope=0.01), p=0.5, training=self.training)

		# NOTE: this takes node-level features/"embeddings" and aggregates to graph-level - use for graph-level classification
		out = global_mean_pool(x, batch=batch)
		if return_graph_embedding:
			return out

		out = self.fc_block1(out)
		return out