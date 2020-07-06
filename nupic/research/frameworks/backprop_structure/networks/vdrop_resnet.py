#  Copyright (C) 2020, Numenta, Inc. All rights reserved.
#
#  The information and source code contained herein is the
#  exclusive property of Numenta Inc.  No part of this software
#  may be used, reproduced, stored or distributed in any form,
#  without explicit written authorization from Numenta Inc.

import inspect
from functools import partial

from nupic.research.frameworks.backprop_structure.modules import VDropCentralData
from nupic.research.frameworks.pytorch.models.resnets import ResNet


class VDropResNet(ResNet):
    """
    A subclass of ResNet that manages the vdrop_data.
    """

    def __init__(self, conv_layer, linear_layer, z_logvar_init=-10,
                 vdrop_data_class=VDropCentralData, **kwargs):
        # Can't save the vdrop_data on the self until after nn.Module.__init__()
        # has been called.
        vdrop_data = vdrop_data_class(z_logvar_init=z_logvar_init)

        if "vdrop_data" in inspect.signature(conv_layer).parameters:
            conv_layer = partial(conv_layer, vdrop_data=vdrop_data)

        if "vdrop_data" in inspect.signature(linear_layer).parameters:
            linear_layer = partial(linear_layer, vdrop_data=vdrop_data)

        super().__init__(conv_layer=conv_layer, linear_layer=linear_layer,
                         **kwargs)

        vdrop_data.finalize()
        self.vdrop_data = vdrop_data

    def forward(self, *args, **kwargs):
        self.vdrop_data.compute_forward_data()
        ret = super().forward(*args, **kwargs)
        self.vdrop_data.clear_forward_data()
        return ret

    def to(self, *args, **kwargs):
        ret = super().to(*args, **kwargs)
        self.vdrop_data = self.vdrop_data.to(*args, **kwargs)
        return ret


vdrop_resnet18 = partial(VDropResNet, depth=18)
vdrop_resnet50 = partial(VDropResNet, depth=50)

__all__ = [
    "VDropResNet",
    "vdrop_resnet18",
    "vdrop_resnet50",
]
