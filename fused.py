from torch import nn
from torch_geometric.nn import GCNConv
import torch
from til_branch import swin_encoder
from mrna_branch import gcn_encoder
from cofusion import SelfAttention
import torch.nn.functional as F
from torch.nn import Parameter

class fused_net(nn.Module):
    def __init__(self):
        super(fused_net, self).__init__()
        self.img_encoder = swin_encoder()
        self.mrna_encoder = gcn_encoder(in_channels=8, hidden_channels=16)
        self.self_attention = SelfAttention(32)
        self.linear_hazard1 = nn.Linear(32, 3)
        self.linear_category1 = nn.Linear(32, 3)

        self.linear_squeeze = nn.Linear(3, 1)

        self.linear_hazard3 = nn.Linear(4, 1)
        self.linear_category3 = nn.Linear(4, 3)
        self.linear_hazard1=nn.Linear(32,3)
        self.linear_category1=nn.Linear(32,3)
        self.linear_squeeze=nn.Linear(3,1)
        self.linear_hazard2=nn.Linear(4,1)
        self.linear_category2=nn.Linear(4,3)
        self.activation = nn.ReLU()
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)
    def forward(self, wsi_img, mrna_data, edge_index):
        img_features = self.img_encoder(wsi_img)
        mrna_features = self.mrna_encoder(mrna_data, edge_index)
        enhanced_feature = self.self_attention(img_features, mrna_features)

        hazard1=self.linear_hazard1(enhanced_feature)
        category1=self.linear_category1(enhanced_feature)
        hazard_squeeze=self.linear_squeeze(hazard1)
        category_squeeze=self.linear_squeeze(category1)
        concat1=torch.cat((hazard1,category_squeeze),dim=1)
        concat2=torch.cat((hazard_squeeze,category1),dim=1)

        hazard2 =self.linear_hazard2(concat1)
        hazard2 = hazard2 * self.output_range + self.output_shift
        # hazard2=self.linear_hazard2(concat1)
        category2=self.linear_category2(concat2)
        return img_features,enhanced_feature,hazard2,category2



