#  Copyright (C) 2020, Numenta, Inc. All rights reserved.
#
#  The information and source code contained herein is the
#  exclusive property of Numenta Inc.  No part of this software
#  may be used, reproduced, stored or distributed in any form,
#  without explicit written authorization from Numenta Inc.

import nupic.hardware.frameworks.xilinx.key_parameters.r2_params as r2
from nupic.hardware.frameworks.pytorch.modules import (
    blocksparse_conv2d,
    blocksparse_linear,
)
from nupic.research.frameworks.pytorch.models.resnets import resnet50
# from nupic.research.frameworks.pytorch.models.sparse_resnets import resnet50
from nupic.research.frameworks.pytorch.modules import relu_maybe_kwinners2d


from nupic.research.frameworks.pytorch.models.resnets import resnet50
# from nupic.research.frameworks.pytorch.models.sparse_resnets import resnet50
import torch
config=dict(
    num_classes=100,
    conv_layer=blocksparse_conv2d,
    conv_args=r2.conv_args(),
    linear_layer=blocksparse_linear,
    linear_args=r2.linear_args(),
    act_layer=relu_maybe_kwinners2d,
    act_args=r2.activation_args(),
    norm_args=dict(momentum=0.012),
)

# works with and without the activation layesr

if __name__ == "__main__":
    model = resnet50(**config)
    # device = torch.device("cpu")
    model.forward(torch.rand(10,3,224,224))
    print("Forward pass ok")
