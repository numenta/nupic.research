import torch
import torch.nn as nn
import torch.nn.functional as F

from nupic.research.frameworks.greedy_infomax.models.BilinearInfo import BilinearInfo
from nupic.research.frameworks.greedy_infomax.utils import model_utils


class BlockModel(nn.Module):
    def __init__(self,
                 modules,
                 patch_size=16,
                 overlap=2):
        super(BlockModel, self).__init__()
        self.modules = modules
        self.patch_size = patch_size
        self.overlap = overlap

    # the forward method only emits BilinearInfo estimations
    # notice how there are no detach() calls in this forward pass: the detaching is
    # done by the separate GradientBlock layers, which gives users more flexibility to
    # place them as desired. Maybe you want each layer to receive gradients from
    # multiple BilinearInfo estimators, for example.
    def forward(self, x):
        # Patchify inputs
        x, n_patches_x, n_patches_y = model_utils.patchify_inputs(
            x, self.patch_size, self.overlap
        )
        log_f_module_list = []
        for module in self.modules:
            if isinstance(module, BilinearInfo):
                out = F.adaptive_avg_pool2d(x, 1)
                out = out.reshape(-1, n_patches_x, n_patches_y, out.shape[1])
                out = out.permute(0, 3, 1, 2).contiguous()
                log_f_list = module(out)
                log_f_module_list.append(log_f_list)
            else:
                x = module(x)
        return log_f_module_list

    def encode(self, x):
        # Patchify inputs
        x, n_patches_x, n_patches_y = model_utils.patchify_inputs(
            x, self.patch_size, self.overlap
        )
        all_outputs = []
        for module in self.modules:
            # no need to detach between modules as .encode() will only be called
            # under a torch.no_grad() scope
            x, out = module.encode(x, n_patches_x, n_patches_y)
        # Return patch-level representation from the last block
        return out
