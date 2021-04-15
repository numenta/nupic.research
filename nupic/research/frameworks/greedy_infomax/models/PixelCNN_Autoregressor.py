import torch
import torch.nn as nn

from .PixelCNN import PixelCNNGatedLayer, PixelCNNGatedStack
from nupic.research.frameworks.greedy_infomax.utils import model_utils


class PixelCNN_Autoregressor(torch.nn.Module):
    def __init__(self, in_channels, pixelcnn_layers=4, calc_loss=True,
                 weight_init=True, **kwargs):
        super().__init__()
        self.calc_loss = calc_loss

        layer_objs = [
            PixelCNNGatedLayer.primary(
                in_channels, in_channels, 3, mask_mode="only_vert", **kwargs
            )
        ]
        layer_objs = layer_objs + [
            PixelCNNGatedLayer.secondary(
                in_channels, in_channels, 3, mask_mode="only_vert", **kwargs
            )
            for _ in range(1, pixelcnn_layers)
        ]

        self.stack = PixelCNNGatedStack(*layer_objs)
        self.stack_out = nn.Conv2d(in_channels, in_channels, 1)


        if weight_init:
            self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d,)):
                if m is self.stack_out:
                    # nn.init.kaiming_normal_(
                    #     m.weight, mode="fan_in", nonlinearity="relu"
                    # )
                    model_utils.makeDeltaOrthogonal(
                        m.weight, nn.init.calculate_gain("relu")
                    )
                else:
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_in", nonlinearity="tanh"
                    )
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d)):
                m.momentum = 0.3

    def forward(self, input):
        _, c_out, _ = self.stack(input, input)  # Bc, C, H, W
        c_out = self.stack_out(c_out)

        assert c_out.shape[1] == input.shape[1]

        if self.calc_loss:
            loss = self.loss(input, c_out)
        else:
            loss = None

        return c_out, loss
