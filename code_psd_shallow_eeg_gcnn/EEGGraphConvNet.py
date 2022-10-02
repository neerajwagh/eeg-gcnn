import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool, BatchNorm

class EEGGraphConvNet(nn.Module):
	def __init__(self, reduced_sensors, sfreq=None, batch_size=32):
		super(EEGGraphConvNet, self).__init__()
		
		# need these for train_model_and_visualize() function
		self.sfreq = sfreq
		self.batch_size = batch_size
		self.input_size = 8 if reduced_sensors else 62

		# layers
		self.conv1 = GCNConv(6, 32, improved=True, cached=True, normalize=False)
		self.conv2 = GCNConv(32, 20, improved=True, cached=True, normalize=False)
		self.conv2_bn = BatchNorm(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

		self.fc_block1 = nn.Linear(20, 10)
		self.fc_block2 = nn.Linear(10, 2)
		# Xavier initializations  #init gcn layers
		self.fc_block1.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if type(x) == nn.Linear else None)
		self.fc_block2.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if type(x) == nn.Linear else None)

		# Xavier initializations
		self.fc_block1.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if type(x) == nn.Linear else None)

	# NOTE: adapted from https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#learning-methods-on-graphs
	def forward(self, x, edge_index, edge_weight, batch, return_graph_embedding=False):

		x = F.leaky_relu(self.conv1(x, edge_index, edge_weight=edge_weight))
		x = F.leaky_relu(self.conv2_bn(self.conv2(x, edge_index, edge_weight=edge_weight)))

		# NOTE: this takes node-level features/"embeddings" and aggregates to graph-level - use for graph-level classification

		out = global_add_pool(x, batch=batch)
		if return_graph_embedding:
			return out
		out = F.dropout(out, p=0.2, training=self.training)
		out = F.leaky_relu(self.fc_block1(out))
		out = self.fc_block2(out)
		return out