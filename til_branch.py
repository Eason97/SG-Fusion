import torch
import torch.nn as nn
from swinnet import SwinTIL
from swinv2_net import SwinTransformerV2

class swin_encoder(nn.Module):
    def __init__(self):
        super(swin_encoder, self).__init__()
        self.TIL_features = SwinTIL(SwinTransformerV2())

        # Define the classification branch
        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True)
        )


    def forward(self, til):
        til_features = self.TIL_features(til)
        feature_size = int(til_features.size(2) ** 0.5)
        reshaped_tensor = til_features.view(til_features.size(0), 512, feature_size, feature_size)
        x = self.avgpool(reshaped_tensor)
        x = x.reshape(x.size(0), -1)
        x = self.fc_layers(x)
        return x



