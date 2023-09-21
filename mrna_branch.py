from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import math
import torch
from blocks.encoder_layer import EncoderLayer
class Encoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, n_layers, drop_prob):
        super(Encoder, self).__init__()
        self.linear = nn.Linear(1, 8)
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])
    def forward(self, x):
        value2vector = self.linear(x)
        for layer in self.layers:
            x = layer(value2vector)
        return x

class gcn_encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(gcn_encoder, self).__init__()
        self.transformer = Encoder(8, 8, 1, 4, 0)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(5724 * hidden_channels, 32)  # 添加线性层

    def forward(self, x, edge_index):
        bs = x.shape[0]
        x = self.transformer(x)
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        gcn_features = x.view(bs, -1)
        output = self.fc(gcn_features)
        return output


