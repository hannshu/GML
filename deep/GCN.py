import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):

    def __init__(self, inFeatures, out) -> None:
        super().__init__()

        self.layer1 = GCNConv(inFeatures, inFeatures, bias=False)
        self.layer2 = GCNConv(inFeatures, out, bias=False)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.layer1(x, edge_index)
        x = self.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.layer2(x, edge_index)
        return self.softmax(x)
