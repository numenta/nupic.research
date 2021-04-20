import torch.nn as nn
import torch.nn.functional as F

from nupic.research.frameworks.greedy_infomax.models.BilinearInfo import \
    BilinearInfo
from nupic.research.frameworks.greedy_infomax.utils import model_utils


class PreActBlockNoBN(nn.Module):
    """Pre-activation version of the BasicBlock."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlockNoBN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1
        )
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes, kernel_size=1, stride=stride
                )
            )

    def forward(self, x):
        out = F.relu(x)
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += shortcut
        return out


class PreActBottleneckNoBN(nn.Module):
    """Pre-activation version of the original Bottleneck module."""

    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneckNoBN, self).__init__()
        # self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        # self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes, kernel_size=1, stride=stride
                )
            )

    def forward(self, x):
        out = F.relu(x)
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(out))
        out = self.conv3(F.relu(out))
        out += shortcut
        return out


class ResNet_Encoder(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        filter,
        encoder_num,
        negative_samples = 16,
        k_predictions = 5,
        patch_size=16,
        input_dims=3,
        weight_init = True,
    ):
        super(ResNet_Encoder, self).__init__()
        self.encoder_num = encoder_num

        self.overlap = 2

        self.patch_size = patch_size
        self.filter = filter

        self.model = nn.Sequential()

        if encoder_num == 0:
            self.model.add_module(
                "Conv1",
                nn.Conv2d(
                    input_dims, self.filter[0], kernel_size=5, stride=1, padding=2
                ),
            )
            self.in_planes = self.filter[0]
            self.first_stride = 1
        elif encoder_num > 2:
            self.in_planes = self.filter[0] * block.expansion
            self.first_stride = 2
        else:
            self.in_planes = (self.filter[0] // 2) * block.expansion
            self.first_stride = 2

        for idx in range(len(num_blocks)):
            self.model.add_module(
                "layer {}".format((idx)),
                self._make_layer(
                    block, self.filter[idx], num_blocks[idx], stride=self.first_stride
                ),
            )
            self.first_stride = 2

        self.bilinear_model = BilinearInfo(
                in_channels=self.in_planes,
                out_channels=self.in_planes,
                negative_samples = negative_samples,
                k_predictions = k_predictions,
            )

        if weight_init:
            self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d,)):
                model_utils.makeDeltaOrthogonal(
                    m.weight, nn.init.calculate_gain("relu")
                )
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d)):
                m.momentum = 0.3

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def forward(self, x, n_patches_x, n_patches_y):
        z = self.model(x)
        out = F.adaptive_avg_pool2d(z, 1)
        out = out.reshape(-1, n_patches_x, n_patches_y, out.shape[1])
        out = out.permute(0, 3, 1, 2).contiguous()
        log_fk, true_f = self.bilinear_model(out, out)
        return log_fk, true_f, z

    def encode(self, x, n_patches_x, n_patches_y):
        z = self.model(x)
        out = F.adaptive_avg_pool2d(z, 1)
        out = out.reshape(-1, n_patches_x, n_patches_y, out.shape[1])
        out = out.permute(0, 3, 1, 2).contiguous()
        representation = F.adaptive_avg_pool2d(out, 1).reshape(out.shape[0], -1)
        return representation, z
