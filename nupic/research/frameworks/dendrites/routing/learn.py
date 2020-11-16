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

"""
Perform the routing task with a dendrite layer by either (a) learning just the
dendrite weights, or (b) learning both the feed-forward and dendrite weights together
"""

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


def init_test_scenario(
    mode,
    dim_in,
    dim_out,
    num_contexts,
    dim_context,
    dendrite_module
):
    """
    Returns the routing function, dendrite layer, context vectors, and device to use
    in the "learn to route" experiment

    :param mode: must be one of ("dendrites", "all")
                 "dendrites" -> learn only dendrite weights while setting feed-forward
                 weights to those of the routing function
                 "all" -> learn both feed-forward and dendrite weights
    :param dim_in: the number of dimensions in the input to the routing function and
                   test module
    :param dim_out: the number of dimensions in the sparse linear output of the routing
                    function and test layer
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
        dim_in=dim_in,
        dim_out=dim_out,
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

    # If only training the dendrite weights, initialize the dendrite module using the
    # same feed-forward sparse weights as the routing function, otherwise if learning
    # feed-forward weights, use `torch.nn.Linear`

    # Also, note that the value passed to `dendrite_sparsity` is irrelevant since the
    # context weights are subsequently
    # overwritten
    if mode == "dendrites":
        dendrite_layer_forward_module = r.sparse_weights.module
    elif mode == "all":
        dendrite_layer_forward_module = torch.nn.Linear(dim_in, dim_out, bias=False)
    else:
        raise Exception("Invalid value for `mode`: {}".format(mode))

    dendrite_layer = dendrite_module(
        module=dendrite_layer_forward_module,
        num_segments=num_contexts,
        dim_context=dim_context,
        module_sparsity=0.7,
        dendrite_sparsity=0.0
    )

    # In this version of learning to route, there is no sparsity constraint on the
    # dendrite weights
    dendrite_layer.register_buffer(
        "zero_mask",
        torch.ones(dendrite_layer.zero_mask.shape).half()
    )

    # Place objects that inherit from torch.nn.Module on device
    r = r.to(device)
    dendrite_layer = dendrite_layer.to(device)
    context_vectors = context_vectors.to(device)

    return r, dendrite_layer, context_vectors, device


def init_dataloader(
    routing_function,
    context_vectors,
    device,
    batch_size,
    x_min,
    x_max
):
    """
    Returns a torch DataLoader for the routing task given a random routing function

    :param routing_function: the random routing function
    :param context_vectors: 2D torch Tensor in which each row gives a context vector
    :param device: device to use ('cpu' or 'cuda')
    :param batch_size: the batch size during training and evaluation
    :param x_min: the minimum bound of the uniform distribution from which input
                  vectors are i.i.d. sampled along each input dimension
    :param x_max: the maximum bound of the uniform distribution from which input
                  vectors are i.i.d. sampled along each input dimension
    """
    routing_test_dataset = RoutingDataset(
        routing_function=routing_function,
        input_size=routing_function.sparse_weights.module.in_features,
        context_vectors=context_vectors,
        device=device,
        x_min=x_min,
        x_max=x_max,
    )

    routing_test_dataloader = DataLoader(
        dataset=routing_test_dataset,
        batch_size=batch_size
    )

    return routing_test_dataloader


def init_optimizer(layer, mode):
    if mode == "dendrites":
        return torch.optim.SGD(layer.parameters(), lr=0.5)
    elif mode == "all":
        return torch.optim.Adam(layer.parameters(), lr=1e-5)


def learn_to_route(
    mode,
    dim_in,
    dim_out,
    num_contexts,
    dim_context,
    dendrite_module,
    batch_size=64,
    num_training_epochs=5000
):
    """
    Trains a dendrite layer to match an arbitrary routing function

    :param mode: must be one of ("dendrites", "all")
                 "dendrites" -> learn only dendrite weights while setting feed-forward
                 weights to those of the routing function
                 "all" -> learn both feed-forward and dendrite weights
    :param dim_in: the number of dimensions in the input to the routing function and
                   test module
    :param dim_out: the number of dimensions in the sparse linear output of the routing
                    function and test layer
    :param num_contexts: the number of unique random binary vectors in the routing
                         function that can "route" the sparse linear output, and also
                         the number of unique context vectors
    :param dim_context: the number of dimensions in the context vectors
    :param dendrite_module: a torch.nn.Module subclass that implements a dendrite
                            module in addition to a linear feed-forward module
    :param batch_size: the batch size during training and evaluation
    :param num_training_epochs: the number of epochs for which to train the dendrite
                                layer
    """

    r, dendrite_layer, context_vectors, device = init_test_scenario(
        mode=mode,
        dim_in=dim_in,
        dim_out=dim_out,
        num_contexts=num_contexts,
        dim_context=dim_context,
        dendrite_module=dendrite_module
    )

    train_dataloader = init_dataloader(
        routing_function=r,
        context_vectors=context_vectors,
        device=device,
        batch_size=batch_size,
        x_min=-2.0,
        x_max=2.0,
    )

    test_dataloader = init_dataloader(
        routing_function=r,
        context_vectors=context_vectors,
        device=device,
        batch_size=batch_size,
        x_min=2.0,
        x_max=6.0,
    )

    optimizer = init_optimizer(mode=mode, layer=dendrite_layer)

    print("epoch,mean_loss,mean_abs_err")
    for epoch in range(1, num_training_epochs + 1):

        # Select L1 weight decay penalty based on scenario
        if mode == "dendrites":
            l1_weight_decay = 0.0
        elif mode == "all":
            l1_weight_decay = 1e-6

        train_dendrite_model(
            model=dendrite_layer,
            loader=train_dataloader,
            optimizer=optimizer,
            device=device,
            criterion=F.l1_loss,
            concat=False,
            l1_weight_decay=l1_weight_decay
        )

        # Validate model
        results = evaluate_dendrite_model(
            model=dendrite_layer,
            loader=test_dataloader,
            device=device,
            criterion=F.l1_loss,
            concat=False
        )

        print("{},{}".format(
            epoch, results["mean_abs_err"]
        ))


if __name__ == "__main__":

    # Learn dendrite weights that learn to route, while keeping feedforward weights
    # fixed

    learn_to_route(
        mode="all",
        dim_in=100,
        dim_out=100,
        num_contexts=10,
        dim_context=100,
        dendrite_module=AbsoluteMaxGatingDendriticLayer
    )
