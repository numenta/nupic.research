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
from torch.utils.data import DataLoader
from nupic.research.frameworks.pytorch.model_utils import evaluate_model
import torch.nn.functional as F
from functools import partial
import sys


class NoiseRobustnessTest:
    """
    Replaces the standard evaluate model function with a loop over the same function
    that conducts a noise robustness test on each iteration. In the test, validation
    accuracy is reported for samples with varying percentages of Gaussian
    noise. The output of this mixin is a curve with noise percentage on the x axis
    and validation accuracy on the y axis.
    """
    def setup_experiment(self, config):
        """
        :param config:
            - noise_levels: Optional, a list of floats between 0 and 1 which specify
            how many units of the input image will receive Gaussian noise.
        """
        super().setup_experiment(config)
        noise_levels = config.get("noise_levels", torch.range(0, 1, 0.1))
        default_evaluate_model_func = config.get("evaluate_model_func", evaluate_model)
        self.evaluate_model = partial(
            evaluate_model_with_noise(
                evaluate_model_func=default_evaluate_model_func,
                noise_levels=noise_levels)
        )

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["setup_experiment"].append("NoiseRobustnessTest: Initialize")
        return eo

def evaluate_model_with_noise(model,
                              loader,
                              device,
                              batches_in_epoch=sys.maxsize,
                              criterion=F.nll_loss,
                              complexity_loss_fn=None,
                              active_classes=None,
                              progress=None,
                              post_batch_callback=None,
                              transform_to_device_fn=None,
                              evaluate_model_func=None,
                              noise_levels=None,):
    all_results = {}
    dataset = loader.dataset
    original_getitem = dataset.__getitem__
    for noise_level in noise_levels:
        dataset.__getitem__ = partial(noisy_getitem(original_getitem, noise_level=noise_level))
        all_results[noise_level] = evaluate_model_func(
            model,
            loader,
            device,
            batches_in_epoch=batches_in_epoch,
            criterion=criterion,
              complexity_loss_fn=complexity_loss_fn,
              active_classes=active_classes,
              progress=progress,
              post_batch_callback=post_batch_callback,
              transform_to_device_fn=transform_to_device_fn,
        )

def noisy_getitem(idx, original_getitem, noise_level=0):
    data, target = original_getitem(idx)
    noise_indices = torch.bernoulli(torch.ones_like(data)*noise_level)
    noise = torch.randn_like(data) * noise_indices
    return data + noise, target