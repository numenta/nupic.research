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


def generate_context_vectors(num_contexts, n_dim, percent_on=0.2):
    """
    Returns a binary Torch tensor of shape (num_vectors, n_dim) where each row
    represents a context vector with n_dim * percent_on non-zero values
    :param num_contexts: the number of context vectors
    :type num_contexts: int
    :param n_dim: the size of each binary vector
    :type n_dim: int
    :param percent_on: the fraction of non-zero
    :type percent_on: float
    """
    num_ones = int(percent_on * n_dim)
    num_zeros = n_dim - num_ones

    context_vectors = torch.cat((
        torch.zeros((num_contexts, num_zeros)),
        torch.ones((num_contexts, num_ones))
    ), dim=1)

    # All rows in context_vectors are currently the same; they need to be shuffled
    for i in range(num_contexts):
        context_vectors[i, :] = context_vectors[i, torch.randperm(n_dim)]

    return context_vectors


def generate_random_binary_vectors(k, n_dim, sparsity_level=0.5):
    """
    Returns a Torch tensor of shape (k, n_dim) where each each entry is 1
    with probability (1 - sparsity_level), 0 otherwise

    :param k: the number of unique binary vectors
    :type k: int
    :param n_dim: the size of each binary vector
    :type n_dim: int
    :param sparsity_level: the expected level of sparsity of each binary vector
    :type n_dim: float
    """
    binary_vectors = torch.rand((k, n_dim))
    binary_vectors = torch.where(
        binary_vectors > sparsity_level,
        torch.zeros((k, n_dim)),
        torch.ones((k, n_dim))
    )
    return binary_vectors


def get_gating_context_weights(output_masks, context_vectors, num_dendrites):
    """
    Returns a torch Tensor of shape (num_units, num_dendrites, dim_context) that gives
    a near-optimal choice of context/dendrite weights to a sigmoid gating verion of
    routing given any routing function specified by output_masks and arbitrary choice
    of context vectors context_vectors (num_units: number of output units in feed-
    forward module, dim_context: the number of dimensions in each context vector)

    :param output_masks: 2D torch Tensor of binary output masks
    :param context_vectors: 2D torch Tensor of binary context vectors
    :param num_dendrites: the number of dendrite/context weights per unit
    """
    num_units = output_masks.size(1)
    dim_context = context_vectors.size(1)

    # Context weights
    context_weights = torch.zeros((num_units, num_dendrites, dim_context))

    for m in range(num_units):
        for j in range(num_dendrites):
            for c in range(dim_context):
                if context_vectors[j, c] > 0.0:

                    # if output mask j is on for output unit m, then set context weight
                    # to +1
                    if output_masks[j, m] > 0.0:
                        context_weights[m, j, c] = 1.0

                    # otherwise, if output mask j is off for output unit m, then set
                    # context weight to -1
                    else:
                        context_weights[m, j, c] = -1.0

    return context_weights


def train_dendrite_model(model, loader, optimizer, device, criterion):
    """
    Trains a dendritic network model by iterating through all batches in the given
    dataloader

    :param model: a torch.nn.Module subclass that implements a dendrite module in
                  addition to a linear feed-forward module, and takes both feedforward
                  and context inputs to its `forward` method
    :param loader: a torch dataloader that iterates over all train and test batches
    :param optimizer: optimizer object used to train the model
    :param device: device to use ('cpu' or 'cuda')
    :param criterion: loss function to minimize
    """
    model.train()

    for data, context, target in loader:

        # for i in range(target.shape[0]):
        #     print(target[i, :])

        data = data.to(device)
        context = context.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(data, context)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()


def evaluate_dendrite_model(model, loader, device, criterion):
    """
    Evaluates a dendritic network on a specified criterion by iterating through all
    batches in the given dataloader

    :param model: a torch.nn.Module subclass that implements a dendrite module in
                  addition to a linear feed-forward module, and takes both feedforward
                  and context inputs to its `forward` method
    :param loader: a torch dataloader that iterates over all train and test batches
    :param device: device to use ('cpu' or 'cuda')
    :param criterion: loss function to minimize
    """
    model.to(device)
    model.eval()
    loss = torch.tensor(0.0, device=device)

    with torch.no_grad():
        for data, context, target in loader:

            data = data.to(device)
            context = context.to(device)
            target = target.to(device)

            output = model(data, context)
            loss += criterion(output, target, reduction="sum")

    loss = loss.item()
    return {"loss": loss}
