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
# ----------------------------------------------------------------------

import os

import pkg_resources
import torch

from nupic.research.frameworks.vernon.experiments import SupervisedExperiment

__all__ = ["TorchProfilerMixin", "inject_torch_profiler_mixin"]


class TorchProfilerMixin:
    """
    Mixin class enabling profiling via pytorch's native profiler.
    See https://pytorch.org/docs/stable/profiler.html

    .. note::
        Requires pytorch 1.8.1 or higher
    """

    def __init__(self, *args, **kwargs):
        pkg_resources.require("torch>=1.8.1")
        super().__init__(*args, **kwargs)
        self._profiler = None

    def setup_experiment(self, config):
        super().setup_experiment(config)

        # Whether or not to export chrome trace
        self._export_chrome_trace = config.get("export_chrome_trace", False)

        # Default profiler args
        self._profiler_args = config.get("profiler", {
            "with_stack": True,
            "record_shapes": True,
            "schedule": torch.profiler.schedule(wait=1, warmup=1, active=5)
        })

    def train_epoch(self):
        profiler_path = os.path.join(self.logdir, "profiler")
        # Default profiler output to tensorboard.
        # Requires `torch-tb-profiler` tensorboard plugin
        profiler_args = {
            **self._profiler_args,
            "on_trace_ready": torch.profiler.tensorboard_trace_handler(profiler_path)
        }
        with torch.profiler.profile(**profiler_args) as prof:
            self._profiler = prof
            super().train_epoch()

        if self._export_chrome_trace and self._profiler is not None:
            self._profiler.export_chrome_trace(profiler_path)

        self._profiler = None

    def post_batch(self, *args, **kwargs):
        super().post_batch(*args, **kwargs)
        if self._profiler is not None:
            self._profiler.step()

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["setup_experiment"].append("TorchProfilerMixin initialization")
        eo["train_epoch"].insert(0, "TorchProfilerMixin begin")
        eo["post_batch"].append("TorchProfilerMixin step")
        eo["train_epoch"].append("TorchProfilerMixin end")
        return eo


def inject_torch_profiler_mixin(experiment_class):
    """
    Injects torch profiler mixin to the given experiment class
    """
    assert issubclass(experiment_class, SupervisedExperiment)
    return type(
        f"Profile{experiment_class.__name__}", (TorchProfilerMixin, experiment_class),
        {}
    )
