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

from nupic.research.frameworks.pytorch import l1_regularization_step

def generate_context_integers(num_contexts):
    return torch.arange(num_contexts, dtype=torch.float32).reshape(num_contexts, 1)

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
        torch.ones((k, n_dim)),
        torch.zeros((k, n_dim))
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


def train_dendrite_model(
    model,
    loader,
    optimizer,
    device,
    criterion,
    context_model=None,
    concat=False,
    l1_weight_decay=0.0
):
    """
    Trains a regular network model by iterating through all batches in the given
    dataloader

    :param model: a torch.nn.Module subclass that implements a dendrite module in
                  addition to a linear feed-forward module, and takes both feedforward
                  and context inputs to its `forward` method
    :param loader: a torch dataloader that iterates over all train and test batches
    :param optimizer: optimizer object used to train the model
    :param device: device to use ('cpu' or 'cuda')
    :param criterion: loss function to minimize
    :param context_model: a torch.nn.Module subclass which generates context vectors.
                          If the context vector should be learned, then
                          the contexts in the dataset are integers from which a context
                          will be generated.
    :param concat: if True, assumes input and context vectors are concatenated together
                   and model takes just a single input to its `forward`, otherwise
                   assumes input and context vectors are separate and model's `forward`
                   function takes a regular input and contextual input separately
    :param l1_weight_decay: L1 regularization coefficient
    """
    model.train()

    for item in loader:

        optimizer.zero_grad()
        if concat:
            data, target = item

            data = data.to(device)
            target = target.to(device)

            output = model(data)

        else:
            data, context, target = item

            data = data.to(device)
            context = context.to(device)
            if context_model:
                # context model generates a context model from the context input
                context = context_model(context)
            target = target.to(device)

            output = model(data, context)

        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        # Perform L1 weight decay
        if l1_weight_decay > 0.0:
            l1_regularization_step(
                params=model.parameters(),
                lr=optimizer.param_groups[0]["lr"],
                weight_decay=l1_weight_decay
            )


def evaluate_dendrite_model(model, loader, device, criterion, context_model=None, concat=False):
    """
    Evaluates a model on a specified criterion by iterating through all batches in the
    given dataloader, and returns a dict of metrics that give evaluation performance

    :param model: a torch.nn.Module subclass that implements a dendrite module in
                  addition to a linear feed-forward module, and takes both feedforward
                  and context inputs to its `forward` method
    :param loader: a torch dataloader that iterates over all train and test batches
    :param device: device to use ('cpu' or 'cuda')
    :param criterion: loss function to minimize
    :param context_model: if not None, a torch.nn.Module which produces a context vector.
    :param concat: if True, assumes input and context vectors are concatenated together
                   and model takes just a single input to its `forward`, otherwise
                   assumes input and context vectors are separate and model's `forward`
                   function takes a regular input and contextual input separately
    """
    model.to(device)
    model.eval()

    loss = torch.tensor(0.0, device=device)
    mean_abs_err = torch.tensor(0.0, device=device)

    with torch.no_grad():
        for item in loader:

            if concat:
                data, target = item

                data = data.to(device)
                target = target.to(device)

                output = model(data)

            else:
                data, context, target = item

                data = data.to(device)
                context = context.to(device)
                if context_model:
                    context = context_model(context)
                target = target.to(device)

                output = model(data, context)

            loss += criterion(output, target, reduction="sum")

            # Report mean absolute error
            abs_err = torch.abs(output - target)
            mean_abs_err += torch.mean(abs_err)

    mean_abs_err = mean_abs_err.item() / len(loader)
    return {
        "loss": loss.item(),
        "mean_abs_err": mean_abs_err
    }
