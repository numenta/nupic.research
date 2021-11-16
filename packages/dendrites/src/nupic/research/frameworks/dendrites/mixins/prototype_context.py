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
    "PrototypeContext",
    "compute_prototype",
    "infer_prototype",
]


class PrototypeContext(metaclass=abc.ABCMeta):
    """
    When training a dendritic network, use the prototype method for computing context
    vectors (that dendrites receive as input) for both training and inference.
    """

    def setup_experiment(self, config):
        # Since the prototype vector is an element-wise mean of individual data samples,
        # it's necessarily the same dimension as the input
        model_args = config.get("model_args")
        dim_context = model_args.get("dim_context")
        input_size = model_args.get("input_size")

        assert dim_context == input_size, ("For prototype experiments `dim_context` "
                                           "must match `input_size`")

        super().setup_experiment(config)

        # Tensor for accumulating each task's prototype vector
        self.contexts = torch.zeros((0, self.model.input_size))
        self.contexts = self.contexts.to(self.device)

    def run_task(self):
        self.train_loader.sampler.set_active_tasks(self.current_task)

        # Construct a context vector by computing the prototype of all training examples
        self.context_vector = compute_prototype(self.train_loader).to(self.device)
        self.contexts = torch.cat((self.contexts, self.context_vector.unsqueeze(0)))
        self.train_context_fn = train_prototype(self.context_vector)

        return super().run_task()

    def validate(self, loader=None):
        if loader is None:
            loader = self.val_loader

        # TODO: take out constants in the call below
        return evaluate_dendrite_model(model=self.model,
                                       loader=loader,
                                       device=self.device,
                                       criterion=self.error_loss,
                                       share_labels=True, num_labels=10,
                                       infer_context_fn=infer_prototype(self.contexts))


def compute_prototype(loader):
    """
    Returns the prototype vector of all samples iterated over in `loader`.
    """
    prototype_vector = torch.zeros([])
    n_prototype = 0
    for x, _ in loader:
        if isinstance(x, list):
            x = x[0]
        x = x.flatten(start_dim=1)
        n_x = x.size(0)

        prototype_vector = prototype_vector + x.sum(dim=0)
        n_prototype += n_x

    prototype_vector /= n_prototype
    return prototype_vector


def train_prototype(context_vector):
    """
    Returns a function that takes a batch of training examples and returns the same
    context vector for each
    """

    def _train_prototype(data):
        context = context_vector.repeat(data.shape[0], 1)
        return context

    return _train_prototype


def infer_prototype(contexts):
    """
    Returns a function that takes a batch of test examples and returns a 2D array where
    row i gives the the prototype vector closest to the ith test example.
    """

    def _infer_prototype(data):
        context = torch.cdist(contexts, data)
        context = context.argmin(dim=0)
        context = contexts[context]
        return context

    return _infer_prototype
