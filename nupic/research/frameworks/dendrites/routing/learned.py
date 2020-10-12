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
import torch.nn.functional as F
from torch.utils.data import DataLoader

from nupic.research.frameworks.dendrites import AbsoluteMaxGatingDendriticLayer
from nupic.research.frameworks.dendrites.routing import (
    RoutingDataset,
    RoutingFunction,
    evaluate_dendrite_model,
    generate_context_vectors,
    train_dendrite_model,
)


def init_test_scenario(dim_in, dim_out, num_contexts, dim_context, dendrite_module):
    """
    Returns the routing function, dendritic network, context vectors, and device to use
    in the "learning to route" experiment

    :param dim_in: the number of dimensions in the input to the routing function and
                   test module
    :param dim_out: the number of dimensions in the sparse linear output of the routing
                    function and test network
    :param num_contexts: the number of unique random binary vectors in the routing
                         function that can "route" the sparse linear output, and also
                         the number of unique context vectors
    :param dim_context: the number of dimensions in the context vectors
    :param dendrite_module: a torch.nn.Module subclass that implements a dendrite
                            module in addition to a linear feed-forward module
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize routing function that this task will try to hardcode, and set
    # `requires_grad=False` since the routing function is static
    r = RoutingFunction(
        d_in=dim_in,
        d_out=dim_out,
        k=num_contexts,
        device=device,
        sparsity=0.7
    )
    r.sparse_weights.module.weight.requires_grad = False

    # Initialize context vectors, where each context vector corresponds to an output
    # mask in the routing function
    context_vectors = generate_context_vectors(
        num_contexts=num_contexts,
        n_dim=dim_context,
        percent_on=0.2
    )

    # Initialize dendrite module using the same feed-forward sparse weights as the
    # routing function; also note that the value passed to `dendrite_sparsity` is
    # irrelevant since the context weights are subsequently overwritten
    dendritic_network = dendrite_module(
        module=r.sparse_weights.module,
        num_segments=num_contexts,
        dim_context=dim_context,
        module_sparsity=0.7,
        dendrite_sparsity=0.5
    )

    # In this version of learning to route, there is no sparsity constraint on the
    # dendritic weights
    dendritic_network.register_buffer(
        "zero_mask",
        torch.ones(dendritic_network.zero_mask.shape).half()
    )

    # Place objects that inherit from torch.nn.Module on device
    r = r.to(device)
    dendritic_network = dendritic_network.to(device)

    return r, dendritic_network, context_vectors, device


def init_dataloader(routing_function, context_vectors, device, batch_size):
    """
    Returns a torch DataLoader for the routing task given a random routing function

    :param routing_function: the random routing function
    :param context_vectors: 2D torch Tensor in which each row gives a context vector
    :param device: device to use ('cpu' or 'cuda')
    :param batch_size: the batch size during training and evaluation
    """
    routing_test_dataset = RoutingDataset(
        routing_function=routing_function,
        input_size=routing_function.sparse_weights.module.in_features,
        context_vectors=context_vectors,
        device=device
    )

    routing_test_dataloader = DataLoader(
        dataset=routing_test_dataset,
        batch_size=batch_size
    )

    return routing_test_dataloader


def init_optimizer(network, lr=0.1):
    return torch.optim.SGD(network.parameters(), lr=lr)


def learn_to_route(
    dim_in,
    dim_out,
    num_contexts,
    dim_context,
    dendrite_module,
    batch_size=64,
    num_training_epochs=1000
):
    """
    Trains a dendritic network to match an arbitrary routing function, while only
    learning the dendritic weights with no sparsity constraint

    :param dim_in: the number of dimensions in the input to the routing function and
                   test module
    :param dim_out: the number of dimensions in the sparse linear output of the routing
                    function and test network
    :param num_contexts: the number of unique random binary vectors in the routing
                         function that can "route" the sparse linear output, and also
                         the number of unique context vectors
    :param dim_context: the number of dimensions in the context vectors
    :param dendrite_module: a torch.nn.Module subclass that implements a dendrite
                            module in addition to a linear feed-forward module
    :param batch_size: the batch size during training and evaluation
    :param num_training_epochs: the number of epochs for which to train the dendritic
                                network
    """

    r, dendritic_network, context_vectors, device = init_test_scenario(
        dim_in=dim_in,
        dim_out=dim_out,
        num_contexts=num_contexts,
        dim_context=dim_context,
        dendrite_module=dendrite_module
    )

    routing_test_dataloader = init_dataloader(
        routing_function=r,
        context_vectors=context_vectors,
        device=device,
        batch_size=batch_size
    )

    optimizer = init_optimizer(network=dendritic_network)

    print("epoch,mean_loss")
    for epoch in range(1, num_training_epochs + 1):

        train_dendrite_model(
            model=dendritic_network,
            loader=routing_test_dataloader,
            optimizer=optimizer,
            device=device,
            criterion=F.mse_loss
        )

        # Validate model - note that we use the same dataset/dataloader for validation
        results = evaluate_dendrite_model(
            model=dendritic_network,
            loader=routing_test_dataloader,
            device=device,
            criterion=F.mse_loss
        )

        print("{},{}".format(
            epoch, results["loss"]
        ))


if __name__ == "__main__":

    # Learn dendritic weights that learn to route, while keeping feedforward weights
    # fixed

    learn_to_route(
        dim_in=100,
        dim_out=100,
        num_contexts=10,
        dim_context=100,
        dendrite_module=AbsoluteMaxGatingDendriticLayer
    )
