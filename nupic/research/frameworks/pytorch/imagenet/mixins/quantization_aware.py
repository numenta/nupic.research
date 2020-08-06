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

# from nupic.research.frameworks.pytorch.imagenet.network_utils import create_model


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

        # extra variables
        self.quantize_weights_per_channel = config.get("quantize_weights_per_channel", True)
        # update model args with fuse relu
        if "config" in config["model_args"]:
            config["model_args"]["config"]["fuse_relu"] = config.get("fuse_relu", True)
        else:
            model_args_config = dict(fuse_relu=config.get("fuse_relu", True))
            config["model_args"]["config"] = model_args_config

        print(config["model_args"])

        super().setup_experiment(config)

        # make an internal copy without data distributed parallel
        # required to save quantized model later
         # self.model_without_ddp = self.model.module

    def transform_model(self):
        """Prepare model for quantization"""

        # prepare model for qat
        _prepare_for_qat(self.model, self.quantize_weights_per_channel)
        self.model.to(self.device)

        # enable observers and fake quantizers to prepare for training
        self.model.apply(enable_observer)
        self.model.apply(enable_fake_quant)
        self.observer_disabled = False
        self.batch_norm_frozen = False

        # DEBUG
        # for name, module in self.model.named_modules():
        #     if not(
        #             isinstance(module, _ObserverBase)
        #             or isinstance(module, FakeQuantize)
        #             or isinstance(module, Identity)
        #     ):
        #         module.register_forward_pre_hook(debug_pre_fwd(name, module.__class__))
        #         module.register_forward_hook(debug_post_fwd(name, module.__class__))

    def pre_batch(self, model, batch_idx):

        # freeze observer parameters for the last 500 batches of last epoch
        if (not self.observer_disabled
            and self.current_epoch == self.epochs
            and self.batch_idx > (.95 * self.total_batches)):
                self.logger.info(f"Freezing observer at epoch {self.current_epoch} and batch {self.batch_idx}")
                self.model.apply(disable_observer)
                self.observer_disabled = True

        # freeze BN parameters for the last 200 batches of last epoch
        if (not self.batch_norm_frozen
            and self.current_epoch == self.epochs
            and self.batch_idx > (.98 * self.total_batches)):
                self.logger.info(f"Freezing BN parameters at epoch {self.current_epoch} and batch {self.batch_idx}")
                self.model.apply(nni.qat.freeze_bn_stats)
                self.batch_norm_frozen = True

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["setup_experiment"].append("Initialize extra variables for QAT")
        eo["transform_model"].append("Prepare model for Quantization")
        return eo

def _prepare_for_qat(model, quantize_weights_per_channel):
    """Prepares model for quantization aware training"""

    # fuse models
    model.fuse_model()

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


def test_script_default_qat(model_class=None):
    """Tests default implementation of QAT network"""
    # create the network
    model = model_class()
    # print(model)
    # import pdb; pdb.set_trace()

    # quantize the network
    device = torch.device("cuda")
    model.fuse_model()
    qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
    model.qconfig = qconfig
    torch.quantization.prepare_qat(model, inplace=True)
    model.to(device)

    # prepare for training
    model.apply(enable_observer)
    model.apply(enable_fake_quant)

    # do a forward pass
    for i in range(10000):
        output = model(torch.randn(32, 3, 224, 224, device=device))
        print(f"Finished {i}")

def test_script_qat_create_model():
    """Tests default implementation of QAT network
       using create model function"""

    device = torch.device("cuda")
    # create the network
    model = create_model(
        model_class=sparse_resnet50,
        model_args={},
        init_batch_norm=False,
        device=device,
        checkpoint_file=None
    )
    # the issue is that I'm sending the model to device
    # before quantizing
    # it has to be done after

    # quantize the network
    model.fuse_model()
    qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
    model.qconfig = qconfig
    torch.quantization.prepare_qat(model, inplace=True)
    model.to(device)

    # prepare for training
    model.apply(enable_observer)
    model.apply(enable_fake_quant)

    # do a forward pass
    model(torch.randn(20, 3, 224, 224, device=device))


if __name__ == "__main__":
    # test_script_default_qat(model_class=resnet50)
    # regular resnet runs with no errors

    for _ in range(3):

        # sparse dense works
        test_script_default_qat(model_class=sparse_resnet50)
        print("Sparse Resnet worked")

        # # regular dense works
        # try:
        #     test_script_default_qat(model_class=resnet50)
        #     print("Dense Resnet worked")
        # except:
        #     print("Dense Resnet failed")

        # # regular dense works
        # try:
        #     test_script_qat_create_model()
        #     print("Create model script worked")
        # except Exception as e:
        #     print("Create model script failed")
        #     print(e)





    # sparse resnet? doesn't work, so the issue is with sparse resnet
    # need to investigate
    # sparse resnet with create model?

"""
default_weight_fake_quant = FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=-128, quant_max=127,
                                                   dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, reduce_range=False)
"""
