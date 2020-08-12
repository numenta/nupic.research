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
import functools

from torch.quantization.quantize import _propagate_qconfig_helper, add_observer_
from torch.quantization import (
    DEFAULT_QAT_MODULE_MAPPING,
    DEFAULT_QCONFIG_PROPAGATE_WHITE_LIST,
    convert,
    disable_observer,
    enable_observer,
    enable_fake_quant,
    FakeQuantize,
    QConfig,
    default_weight_fake_quant,
    MovingAverageMinMaxObserver,
)

from torch.quantization.observer import _ObserverBase
from torch.nn.modules.linear import Identity
from nupic.hardware.frameworks.quantization import QATKWINNER_MODULE_MAPPING

QAT_QUANTIZED_MODULE_MAPPING = dict(DEFAULT_QAT_MODULE_MAPPING)
QAT_QUANTIZED_MODULE_MAPPING.update(QATKWINNER_MODULE_MAPPING)
QAT_QCONFIG_PROPAGATE_WHITE_LIST = (
    set(DEFAULT_QCONFIG_PROPAGATE_WHITE_LIST)
    | set(QATKWINNER_MODULE_MAPPING.keys())
)
from nupic.research.frameworks.pytorch.models.resnets import resnet50
from nupic.research.frameworks.pytorch.models.sparse_resnets import resnet50 as sparse_resnet50


class QuantizationAware(object):
    """
    Trains the network using pytorch quantization aware training.
    Reference for distributed model:
    https://github.com/pytorch/vision/blob/7d8581812792a4a9f5df65cef3d75cea0fe9a954/references/classification/train_quantization.py
    """
    def setup_experiment(self, config):
        """
        Setup experiment for quantization
        Add following variables to config

        :param config: Dictionary containing the configuration parameters

            - quantize_weights_per_channel: Whether to quantize weights per channel.
                                            Defaults to True. If False, quantizer per
                                            tensor
            - when_disable_observers: Determiners At which point during the last epoch to disable
                                      observers. Float from 0 to 1, 0 being the first batch
                                      and 1 being the last batch
                                      Defaults to .95
            - when_disable_batch_norm: Determiners At which point during the last epoch to disable
                                       batch norm. Float from 0 to 1, 0 being the first batch
                                       and 1 being the last batch
                                       Defaults to .98
        """

        # extra variables
        self.quantize_weights_per_channel = config.get("quantize_weights_per_channel", True)
        self.when_disable_observers = config.get("when_disable_observers", .95)
        self.when_freeze_batch_norm = config.get("when_freeze_batch_norm", .98)
        self.fuse_relu = config.get("fuse_relu", False)

        super().setup_experiment(config)

    def transform_model(self):
        """Prepare model for quantization"""

        # prepare model for qat
        _prepare_for_qat(self.model, self.quantize_weights_per_channel, self.fuse_relu)
        self.model.to(self.device)

        # enable observers and fake quantizers to prepare for training
        self.model.apply(enable_observer)
        self.model.apply(enable_fake_quant)
        self.observer_disabled = False
        self.batch_norm_frozen = False

    def pre_batch(self, model, batch_idx):

        # freeze observer parameters for the last 500 batches of last epoch
        if (not self.observer_disabled
            and self.current_epoch == self.epochs
            and self.batch_idx > (self.when_disable_observers * self.total_batches)):
                self.logger.info(f"Freezing observer at epoch {self.current_epoch} and batch {self.batch_idx}")
                self.model.apply(disable_observer)
                self.observer_disabled = True

        # freeze BN parameters for the last 200 batches of last epoch
        if (not self.batch_norm_frozen
            and self.current_epoch == self.epochs
            and self.batch_idx > (self.when_freeze_batch_norm * self.total_batches)):
                self.logger.info(f"Freezing BN parameters at epoch {self.current_epoch} and batch {self.batch_idx}")
                self.model.apply(nni.qat.freeze_bn_stats)
                self.batch_norm_frozen = True

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["setup_experiment"].append("Initialize extra variables for QAT")
        eo["transform_model"].append("Prepare model for Quantization")
        return eo

def _prepare_for_qat(model, quantize_weights_per_channel, fuse_relu):
    """Prepares model for quantization aware training"""

    # fuse models
    model.fuse_model(fuse_relu=fuse_relu)

    # set qconfig
    if quantize_weights_per_channel:
        qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
    else:
        print("Quantizating weights per tensor")
        qconfig = QConfig(
            activation=FakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                quant_min=0,
                                                quant_max=255,
                                                reduce_range=True),
            weight=default_weight_fake_quant
        )
    model.qconfig = qconfig

    # equivalent to quantize.prepare, inplace. require for custom white list
    # propagate qconfig and add observers
    _propagate_qconfig_helper(model, qconfig_dict={},
                            white_list=QAT_QCONFIG_PROPAGATE_WHITE_LIST)
    if not any(hasattr(m, 'qconfig') and m.qconfig for m in model.modules()):
        warnings.warn("None of the submodule got qconfig applied. Make sure you "
                    "passed correct configuration through `qconfig_dict` or "
                    "by assigning the `.qconfig` attribute directly on submodules")
    add_observer_(model)

    # convert modules to their QAT versions. should be sent to device after
    convert(model, QAT_QUANTIZED_MODULE_MAPPING, inplace=True)


def debug_pre_fwd(name, class_name):
    """
    Hook is attached to the operations at the computational graph
    One single pytorch module may contain several operations,
    which is why input and output are tuples.
    For debugging we are interested in input to the first operation i[0]

    Usage:
    for name, module in self.model.named_modules():
        module.register_forward_pre_hook(debug_pre_fwd(name, module.__class__))
    """
    def print_internals(m, i):
        nan_ratio = torch.sum(torch.isnan(i[0])).item() / i[0].numel()
        mean, std = 0, 0
        if nan_ratio == 0:
            mean, std = torch.mean(i[0]).item(), torch.std(i[0]).item()

        print(name, f"INPUT nans: {nan_ratio:.2f}, mean: {mean:.8f}, std: {std:.8f}")

    return print_internals

def debug_post_fwd(name, class_name):
    """
    Hook is attached to the operations at the computational graph
    One single pytorch module may contain several operations,
    which is why input and output are tuples.
    For debugging we are interested in the output for the last operation o[-1]

    Usage:
    for name, module in self.model.named_modules():
        module.register_forward_hook(debug_pre_fwd(name, module.__class__))
    """
    def print_internals(m, i, o):
        nan_ratio = torch.sum(torch.isnan(o[-1])).item() / o[-1].numel()
        mean, std = 0, 0
        if nan_ratio == 0:
            mean, std = torch.mean(o[-1]).item(), torch.std(o[-1]).item()

        print(name, f"OUTPUT nans: {nan_ratio:.2f}, mean: {mean:.8f}, std: {std:.8f}")

    return print_internals

