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

import copy

import torch
from numpy.random import randint

from nupic.research.frameworks.dendrites import AbsoluteMaxGatingDendriticLayer
from nupic.research.frameworks.dendrites.routing import (
    RoutingFunction,
    generate_context_vectors,
    get_gating_context_weights,
)


def run_hardcoded_routing_test(
    d_in,
    d_out,
    k,
    d_context,
    dendrite_module,
    context_weights_fn=None,
    batch_size=100,
    verbose=False
):
    """
    Runs the hardcoded routing test for a specific type of dendritic network

    :param d_in: the number of dimensions in the input to the routing function and test
                 module
    :param d_out: the number of dimensions in the sparse linear output of the routing
                  function and test network
    :param k: the number of unique random binary vectors in the routing function that
              can "route" the sparse linear output, and also the number of unique
              context vectors
    :param d_context: the number of dimensions in the context vectors
    :param dendrite_module: a torch.nn.Module subclass that implements a dendrite
                            module in addition to a linear feed-forward module
    :param context_weights_fn: a function that returns a 3D torch Tensor that gives the
                               near-optimal dendrite values for the specified
                               dendrite_module, and has parameters `output_masks`,
                               `context_vectors`, and `num_dendrites`
    :param batch_size: the number of test inputs
    :param verbose: prints the 
    """

    # Initialize routing function that this task will try to hardcode
    r = RoutingFunction(d_in=d_in, d_out=d_out, k=k, sparsity=0.7)

    # Initialize context vectors, where each context vector corresponds to an output
    # mask in the routing function
    context_vectors = generate_context_vectors(
        num_contexts=k,
        n_dim=d_context,
        percent_on=0.2
    )

    # Initialize dendrite module using the same feed-forward sparse weights as the
    # routing function; also note that the value passed to `dendrite_sparsity` is
    # irrelevant since the context weights are subsequently overwritten
    dendritic_network = dendrite_module(
        module=r.sparse_weights.module,
        num_segments=k,
        dim_context=d_context,
        module_sparsity=0.7,
        dendrite_sparsity=0.0
    )

    dendritic_network.register_buffer(
        "zero_mask",
        copy.deepcopy(r.sparse_weights.zero_mask.half())
    )

    # Choose the context weights specifically so that they can gate the outputs of the
    # forward module
    if context_weights_fn is not None:
        dendritic_network.segments.weights.data = context_weights_fn(
            output_masks=r.output_masks, context_vectors=context_vectors,
            num_dendrites=k
        )

    # Sample a random batch of inputs and random batch of context vectors, and perform
    # hardcoded routing test
    x_test = 4.0 * torch.rand((batch_size, d_in)) - 2.0  # sampled i.i.d. from U[-2, 2)
    context_inds_test = randint(low=0, high=k, size=batch_size).tolist()
    context_test = torch.stack(
        [context_vectors[j, :] for j in context_inds_test],
        dim=0
    )

    target = r(context_inds_test, x_test)
    actual = dendritic_network(x_test, context_test)

    if verbose:

        # Print targets and outputs on the first 15 dimensions
        print("")
        print(" Element-wise outputs along the first 15 dimensions:")
        print("")
        print(" {}{}".format("target".ljust(24), "actual".ljust(24)))
        for target_i, actual_i in zip(target[0, :15], actual[0, :15]):
            
            target_i = str(target_i.item()).ljust(24)
            actual_i = str(actual_i.item()).ljust(24)
            
            print(" {}{}".format(target_i, actual_i))
        print(" ...")
        print("")

    # Report mean absolute error
    mean_abs_error = torch.abs(target - actual).mean().item()
    return {"mean_abs_error": mean_abs_error}


if __name__ == "__main__":

    # Run the hardcoded routing test with a dendritic network that uses "dendrites as
    # gating"; this achieves near-zero mean absolute error, which strongly suggests
    # that dendritic networks can successfully route
    d_in = 100
    d_out = 100
    num_contexts = 10
    d_context = 100
    batch_size = 100

    result = run_hardcoded_routing_test(
        d_in=d_in,
        d_out=d_out,
        k=num_contexts,
        d_context=d_context,
        dendrite_module=AbsoluteMaxGatingDendriticLayer,
        context_weights_fn=get_gating_context_weights,
        batch_size=batch_size,
        verbose=True
    )

    print(" Results over {} examples with {} random contexts".format(
        batch_size, num_contexts
    ))
    print(" > Mean absolute error between R(j, x) and output from dendritic network:")
    print(" > \t{}".format(result["mean_abs_error"]))

    # Sample output below

    # Results over 100 examples with 10 random contexts
    # > Mean absolute error between R(j, x) and output from dendritic network:
    # >      0.0006185245583765209
