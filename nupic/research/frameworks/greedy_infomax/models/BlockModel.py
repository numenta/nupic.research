# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
#
# This work was based on the original Greedy InfoMax codebase from Sindy Lowe:
# https://github.com/loeweX/Greedy_InfoMax
# The Greedy InfoMax paper can be found here:
# https://arxiv.org/abs/1905.11786
# ----------------------------------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F

from nupic.research.frameworks.greedy_infomax.models.BilinearInfo import BilinearInfo
from nupic.research.frameworks.greedy_infomax.utils import model_utils
from nupic.research.frameworks.greedy_infomax.models.UtilityLayers import EmitEncoding



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
    # done by the GradientBlock layers, which gives users more flexibility to
    # place them as desired. Maybe you want each layer to receive gradients from
    # multiple BilinearInfoLegacy estimators, for example.
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
                log_f_list = module.estimate_info(out, out)
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
            if isinstance(module, EmitEncoding):
                out = module.encode(x, n_patches_x, n_patches_y)
                all_outputs.append(out)
            else:
                x = module(x)
        return all_outputs