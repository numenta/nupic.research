import torch
import torch.nn as nn

from nupic.research.frameworks.greedy_infomax.models import \
    ResNet_Encoder, PreActBlockNoBN, PreActBottleneckNoBN, PixelCNN_Autoregressor

from nupic.research.frameworks.greedy_infomax.utils import model_utils

class FullVisionModel(torch.nn.Module):
    def __init__(self,
                 negative_samples=16,
                 k_predictions=5,
                 resnet_50=True,
                 grayscale=True,
                 ):
        super().__init__()
        self.negative_samples = negative_samples
        self.k_predictions = k_predictions
        print("Contrasting against ", self.contrastive_samples, " negative samples")

        block_dims = [3, 4, 6]
        num_channels = [64, 128, 256]

        self.encoder = nn.ModuleList([])

        if resnet_50:
            self.block = PreActBottleneckNoBN
        else:
            self.block = PreActBlockNoBN

        if grayscale:
            input_dims = 1
        else:
            input_dims = 3

        for idx in range(len(block_dims)):
            self.encoder.append(
                ResNet_Encoder(
                    self.block,
                    [block_dims[idx]],
                    [num_channels[idx]],
                    idx,
                    input_dims=input_dims,
                    k_predictions=self.k_predictions,
                    negative_samples=self.negative_samples,
                )
            )

    def forward(self, x):
        x, n_patches_x, n_patches_y = \
            model_utils.patchify_inputs(x, self.patch_size, self.overlap)
        log_fk_list, true_f_list = [], []
        for module in self.encoder:
            log_fk, true_f, z = module(x, n_patches_x, n_patches_y)
            log_fk_list.append(log_fk)
            true_f_list.append(true_f)
            # Detach z to make sure no gradients are flowing in between modules
            x = z.detach()
        return log_fk_list, true_f_list

    def encode(self, x):
        x, n_patches_x, n_patches_y = \
            model_utils.patchify_inputs(x, self.patch_size, self.overlap)
        for module in self.encoder:
            representation, z = module.encode(x, n_patches_x, n_patches_y)
            # Detach z to make sure no gradients are flowing in between modules
            x = z.detach()
        return representation
