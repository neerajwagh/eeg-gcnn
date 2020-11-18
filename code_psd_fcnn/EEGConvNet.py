import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGConvNet(nn.Module):
	def __init__(self, reduced_sensors, sfreq=None, batch_size=32):
		super(EEGConvNet, self).__init__()
		
		self.sfreq = sfreq
		self.batch_size = batch_size
		self.input_size = 8 if reduced_sensors else 62

		self.fc_block1 = nn.Linear(48, 64)
		self.batchnorm1 = nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.fc_block2 = nn.Linear(64, 32)
		self.fc_block3 = nn.Linear(32, 2)

		# Xavier initializations
		self.fc_block1.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if type(x) == nn.Linear else None)
		self.fc_block2.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if type(x) == nn.Linear else None)
		self.fc_block3.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if type(x) == nn.Linear else None)
		
	def forward(self, x):
		x = x.reshape(x.size(0), -1)
		x = F.dropout(F.leaky_relu(self.batchnorm1(self.fc_block1(x)), negative_slope=0.01), p=0.4, training=self.training)
		x = F.dropout(F.leaky_relu(self.fc_block2(x), negative_slope=0.01), p=0.5, training=self.training)
		out = self.fc_block3(x)
		return out