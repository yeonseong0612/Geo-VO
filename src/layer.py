import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv

class GeometricGAT(nn.Module):
    def __init__(self, in_channels = 256, hidden_channels = 256, out_channels = 256, heads = 4):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels // heads,
                               heads=heads, edge_dim=1)
        self.conv2 = GATv2Conv(hidden_channels, out_channels, 
                               heads=heads, edge_dim=1, concat=False)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.relu(x)
        x, (edge_index, alpha) = self.conv2(x, edge_index, edge_attr, return_attention_weights=True)
        
        return x, alpha
    
class CorrBlock(nn.Module):
    def __init__(self, num_levels=4, radius = 3):
        super().__init__()
        self.num_levels = num_levels
        self.radius = radius
    def forward(self, fmap1, fmap2):
        corr = torch.matmul(fmap1, fmap2.transpose(0,1))

        weights = torch.softmax(corr / (fmap1.shape[-1]**0.5), dim=-1)

        return weights