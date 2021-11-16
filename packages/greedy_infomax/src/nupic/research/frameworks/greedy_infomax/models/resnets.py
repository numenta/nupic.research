import torch
import torch.nn as nn
import torch.nn.functional as F
from nupic.research.frameworks.greedy_infomax.models.resnet_encoder import \
    SparsePreActBlockNoBN, PreActBlockNoBN


class ResNet7(nn.Sequential):
    def __init__(self,
                 channels=64,):
        super(ResNet7, self).__init__()
        self.conv1 = nn.Conv2d(1, channels, kernel_size=5, stride=1, padding=2)
        self.sparse_preact_1 = PreActBlockNoBN(channels, channels)
        self.sparse_preact_2 = PreActBlockNoBN(channels, channels)
        self.sparse_preact_3 = PreActBlockNoBN(channels, channels)