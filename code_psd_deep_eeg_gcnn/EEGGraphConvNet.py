import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm, global_add_pool


class EEGGraphConvNet(nn.Module):
    def __init__(self, reduced_sensors, sfreq=None, batch_size=32):
        super(EEGGraphConvNet, self).__init__()
		
		# need these for train_model_and_visualize() function
        self.sfreq = sfreq
        self.batch_size = batch_size
        self.input_size = 8 if reduced_sensors else 62

        self.conv1 = GCNConv(6, 16, improved=True, cached=True, normalize=False)
        self.conv2 = GCNConv(16, 32, improved=True, cached=True, normalize=False)
        self.conv3 = GCNConv(32, 64, improved=True, cached=True, normalize=False)
        self.conv4 = GCNConv(64, 50, improved=True, cached=True, normalize=False)
        self.conv4_bn = BatchNorm(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.fc_block1 = nn.Linear(50, 30)
        self.fc_block2 = nn.Linear(30, 20)
        self.fc_block3 = nn.Linear(20, 2)

        # Xavier initializations  #init gcn layers
        self.fc_block1.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if type(x) == nn.Linear else None)
        self.fc_block2.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if type(x) == nn.Linear else None)
        self.fc_block3.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if type(x) == nn.Linear else None)
    
    def forward(self, x, edge_index, edge_weight, batch, return_graph_embedding=False):
        x = F.leaky_relu(self.conv1(x, edge_index, edge_weight))

        x = F.leaky_relu(self.conv2(x, edge_index, edge_weight))

        x = F.leaky_relu(self.conv3(x, edge_index, edge_weight))
        x = F.leaky_relu(self.conv4_bn(self.conv4(x, edge_index, edge_weight)))
        out = global_add_pool(x, batch=batch)
        if return_graph_embedding:
            return out

        out = F.leaky_relu(self.fc_block1(out), negative_slope=0.01)
        out = F.dropout(out, p = 0.2, training=self.training)
        out = F.leaky_relu(self.fc_block2(out), negative_slope=0.01)
        out = self.fc_block3(out)

        return out