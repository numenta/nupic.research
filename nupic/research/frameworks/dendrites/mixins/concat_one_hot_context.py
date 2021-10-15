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

import abc

import torch

from nupic.research.frameworks.dendrites import evaluate_dendrite_model

__all__ = [
    "OneHotContext"
]


class OneHotContext(metaclass=abc.ABCMeta):
    """
    Use this for passing in 1hot context vector (1 hot index represents the curent task)
    to a model. Original use case is to compare MLP with context to dendritic networks,
    since dendritic networks have context automatically. Code is copied from
    centroid_context.py and modified.
    """

    def run_task(self):
        self.train_loader.sampler.set_active_tasks(self.current_task)

        self.context_vector = torch.zeros(self.dim_context)
        self.context_vector[self.current_task] = 1
        self.train_context_fn = train_centroid(self.context_vector)

        return super().run_task()

    def validate(self, loader=None):
        if loader is None:
            loader = self.val_loader

        # Problem: how do you know which task you are on, and
        # therefore how to set the context?
        # Short term fix:
        # Assume eval_per_task is on, so you validate one task at a time
        self.context_vector = torch.zeros(self.dim_context)
        self.context_vector[loader.sampler.active_tasks] = 1

        # TODO: take out constants in the call below
        return evaluate_dendrite_model(model=self.model,
                                       loader=loader,
                                       device=self.device,
                                       criterion=self.error_loss,
                                       share_labels=True, num_labels=10,
                                       infer_context_fn=train_centroid(
                                           self.context_vector))


def train_centroid(context_vector):
    """
    Returns a function that takes a batch of training examples and returns the same
    context vector for each
    """

    def _train_centroid(data):
        context = context_vector.repeat(data.shape[0], 1)
        return context

    return _train_centroid
