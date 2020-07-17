# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
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
# ----------------------------------------------------------------------

import torch
import torch.nn.intrinsic as nni

from torch.quantization.quantize import _propagate_qconfig_helper, add_observer_
from torch.quantization import (
    DEFAULT_QAT_MODULE_MAPPING,
    DEFAULT_QCONFIG_PROPAGATE_WHITE_LIST,
    convert,
    disable_observer,
    enable_observer,
    enable_fake_quant
)
from torch.quantization.observer import _ObserverBase

from nupic.hardware.frameworks.quantization import QKWINNER_MODULE_MAPPING

QUANTIZED_MODULE_MAPPING = dict(DEFAULT_QAT_MODULE_MAPPING)
QUANTIZED_MODULE_MAPPING.update(QKWINNER_MODULE_MAPPING)
QCONFIG_PROPAGATE_WHITE_LIST = (
    set(DEFAULT_QCONFIG_PROPAGATE_WHITE_LIST)
    | set(QKWINNER_MODULE_MAPPING.keys())
)


class QuantizationAware(object):
    """
    Trains the network using pytorch quantization aware training.
    Reference for distributed model:
    https://github.com/pytorch/vision/blob/7d8581812792a4a9f5df65cef3d75cea0fe9a954/references/classification/train_quantization.py
    """
    def setup_experiment(self, config):
        """
        Setup experiment for quantization
        """
        super().setup_experiment(config)

        # make an internal copy without data distributed parallel
        # required to save quantized model later
        self.model_without_ddp = self.model.module

    def transform_model(self):
        """Prepare model for quantization"""

        backend = "fbgemm"
        self.model.fuse_model()
        self.model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
        torch.quantization.prepare_qat(self.model, inplace=True)
        # resend to device, to be sure
        self.model.to(self.device)

        # # fuse model
        # self.model.fuse_model()

        # # define qconfig
        # backend = "fbgemm"
        # self.model.qconfig = torch.quantization.get_default_qat_qconfig(backend)

        # # prepare for QAT

        # # _propagate_qconfig_helper(self.model, qconfig_dict={},
        # #                           white_list=QCONFIG_PROPAGATE_WHITE_LIST)
        # # add_observer_(self.model)
        # # convert(self.model, QUANTIZED_MODULE_MAPPING, inplace=True)

        self.model.apply(enable_observer)
        self.model.apply(enable_fake_quant)

        # attach hooks to debug
        def print_nan_ratio(module, input):
            input = input[0]
            print("In forward hook")
            nan_ratio = torch.sum(torch.isnan(input)).item() / input.numel()
            print(f"Input nans: {nan_ratio:.2f}")
            if nan_ratio == 0:
                mean_value = torch.mean(input).item()
                print(f"Input mean: {mean_value:.2f}")

        # iterate through all modules
        for module in self.model.modules():
            # check if is an observer
            if isinstance(module, _ObserverBase):
                # add hooks
                module.register_forward_pre_hook(print_nan_ratio)


        # add a hook that gives me the name of the module, when the module rans
        # def print_name_func(name):
        #     return lambda: print(name)

        for name, module in self.model.named_modules():
            module.register_forward_pre_hook(lambda m, i: print(name))

    def pre_epoch(self):

        # does it suffice to do at last epoch?
        # will have to explore parameters
        # if self.current_epoch == (self.epochs):
        #     self.logger.info("Freezing parameters")
        #     # freeze quantization parameters
        #     self.model.apply(disable_observer)
        #     # freeze batch norm parameters
        #     self.model.apply(nni.qat.freeze_bn_stats)
        super().pre_epoch()

    def validate(self, loader=None):
        if loader is None:
            loader = self.val_loader

        # create a quantized version for evaluation
        # quantized_model = convert(
        #     # why call eval here as well?
        #     self.model.eval(), QUANTIZED_MODULE_MAPPING, inplace=False
        # )
        # quantized_model.eval()

        return self.evaluate_model(
            model=self.model, # quantized model only in CPU?
            loader=loader,
            device=self.device,
            criterion=self.error_loss,
            complexity_loss_fn=self.complexity_loss,
            batches_in_epoch=self.batches_in_epoch,
            transform_to_device_fn=self.transform_data_to_device,
        )

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["setup_experiment"].append("Knowledge Distillation initialization")
        return eo
