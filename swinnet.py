import torch
import torch.nn as nn
import numpy as np
from swinv2_net import SwinTransformerV2
class SwinTIL(nn.Module):
    def __init__(self, imagenet_model):
        super(SwinTIL, self).__init__()
        checkpoint = torch.load('swinv2_base_patch4_window8_256.pth', map_location='cpu')
        imagenet_model.load_state_dict(checkpoint['model'])
        self.encoder = imagenet_model
    def forward(self, input):
        x, layer_feature = self.encoder(input)  # [0-2]: [4096,128] [1024,256] [256,512]  [3]=x: [64,1024]
        # change dimension
        x, feature1, feature2, feature3 = x.transpose(1, 2), layer_feature[0].transpose(1, 2), layer_feature[
            1].transpose(1, 2), layer_feature[2].transpose(1, 2)
        x = torch.reshape(x, (x.shape[0], x.shape[1], int(np.sqrt(x.shape[2])), int(np.sqrt(x.shape[2]))))
        feature1 = torch.reshape(feature1, (
        feature1.shape[0], feature1.shape[1], int(np.sqrt(feature1.shape[2])), int(np.sqrt(feature1.shape[2]))))
        feature2 = torch.reshape(feature2, (
        feature2.shape[0], feature2.shape[1], int(np.sqrt(feature2.shape[2])), int(np.sqrt(feature2.shape[2]))))
        feature3 = torch.reshape(feature3, (
        feature3.shape[0], feature3.shape[1], int(np.sqrt(feature3.shape[2]))*int(np.sqrt(feature3.shape[2]))))
        return feature3

