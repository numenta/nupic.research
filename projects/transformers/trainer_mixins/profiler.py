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
from transformers import Trainer

__all__ = ["TorchProfilerMixin", "inject_profiler_mixin"]


class TorchProfilerMixin:
    """
    Mixin to HF Trainer class enabling profiling via pytorch's native profiler.
    See ahttps://pytorch.org/docs/stable/profiler.html

    .. note::
        Requires pytorch 1.8.1 or higher
    """

    def __init__(self, *args, **kwargs):
        pkg_resources.require("torch>=1.8.1")
        super().__init__(*args, **kwargs)
        self._profiler = None

    def train(self, *args, **kwargs):

        # Default profiler output to tensorboard.
        # Requires `torch-tb-profiler` tensorboard plugin
        profiler_path = os.path.join(self.args.output_dir, "profiler")
        profiler_args = {
            "on_trace_ready": torch.profiler.tensorboard_trace_handler(profiler_path)
        }

        mixin_args = self.args.trainer_mixin_args
        profiler_args.update(mixin_args.pop("profiler", {}))
        with torch.profiler.profile(**profiler_args) as prof:
            self._profiler = prof
            res = super().train(*args, **kwargs)

        if mixin_args.get("export_chrome_trace", False):
            self._profiler.export_chrome_trace(profiler_path)

        self._profiler = None
        return res

    def training_step(self, model, inputs):
        train_loss = super().training_step(model, inputs)
        if self._profiler is not None:
            self._profiler.step()
        return train_loss


def inject_profiler_mixin(trainer_class):
    """
    Injects torch profiler mixin to the given trainer class
    """
    assert issubclass(trainer_class, Trainer)
    return type(
        f"Profile{trainer_class.__name__}", (TorchProfilerMixin, trainer_class), {}
    )
