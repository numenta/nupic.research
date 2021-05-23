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

from nupic.research.frameworks.dendrites import (
    evaluate_dendrite_model,
    train_dendrite_model,
)

__all__ = [
    "CentroidContext",
    "compute_centroid",
    "infer_centroid",
]


class CentroidContext(metaclass=abc.ABCMeta):
    """
    When training a dendritic network, use the centroid method for computing context
    vectors (that dendrites receive as input) for both training and inference.
    """

    def setup_experiment(self, config):
        # Since the centroid vector is an element-wise mean of individual data samples,
        # it's necessarily the same dimension as the input
        model_args = config.get("model_args")
        dim_context = model_args.get("dim_context")
        input_size = model_args.get("input_size")

        assert dim_context == input_size, ("For centroid experiments `dim_context` "
                                           "must match `input_size`")

        super().setup_experiment(config)

        # Store batch size
        self.batch_size = config.get("batch_size", 1)

        # Tensor for accumulating each task's centroid vector
        self.contexts = torch.zeros((0, self.model.input_size))
        self.contexts = self.contexts.to(self.device)

        # The following will point to the the 'active' context vector used to train on
        # the current task
        self.context_vector = None

    def run_task(self):
        self.train_loader.sampler.set_active_tasks(self.current_task)

        # Construct a context vector by computing the centroid of all training examples
        self.context_vector = compute_centroid(self.train_loader).to(self.device)
        self.contexts = torch.cat((self.contexts, self.context_vector.unsqueeze(0)))

        return super().run_task()

    def train_epoch(self):
        # TODO: take out constants in the call below. How do we determine num_labels?
        train_dendrite_model(
            model=self.model,
            loader=self.train_loader,
            optimizer=self.optimizer,
            device=self.device,
            criterion=self.error_loss,
            share_labels=True,
            num_labels=10,
            context_vector=self.context_vector,
            post_batch_callback=self.post_batch_wrapper,
        )

    def validate(self, loader=None):
        if loader is None:
            loader = self.val_loader

        # TODO: take out constants in the call below
        return evaluate_dendrite_model(model=self.model,
                                       loader=loader,
                                       device=self.device,
                                       criterion=self.error_loss,
                                       share_labels=True, num_labels=10,
                                       infer_context_fn=infer_centroid(self.contexts))


def compute_centroid(loader):
    """
    Returns the centroid vector of all samples iterated over in `loader`.
    """
    centroid_vector = torch.zeros([])
    n_centroid = 0
    for x, _ in loader:
        if isinstance(x, list):
            x = x[0]
        x = x.flatten(start_dim=1)
        n_x = x.size(0)

        centroid_vector = centroid_vector + x.sum(dim=0)
        n_centroid += n_x

    centroid_vector /= n_centroid
    return centroid_vector


def infer_centroid(contexts):
    """
    Returns a function that takes a batch of test examples and returns a 2D array where
    row i gives the the centroid vector closest to the ith test example.
    """

    def _infer_centroid(data):
        context = torch.cdist(contexts, data)
        context = context.argmin(dim=0)
        context = contexts[context]
        return context

    return _infer_centroid
