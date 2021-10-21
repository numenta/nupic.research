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


import torch

from nupic.research.frameworks.pytorch.datasets import ContextDependentPermutedMNIST

num_tasks = 5
seed = 666
train = True
download = True
dim_context = 1024

sparse_binary_tuple_mnist = ContextDependentPermutedMNIST(
    num_tasks=num_tasks,
    seed=seed,
    train=train,
    download=False,
    dim_context=dim_context,
)

data0, target0 = sparse_binary_tuple_mnist[0]
data1, target1 = sparse_binary_tuple_mnist[65_432]  # second task, first permutation

assert len(sparse_binary_tuple_mnist) == 60_000 * num_tasks
assert len(sparse_binary_tuple_mnist.data) == 60_000
assert isinstance(data0, tuple)
assert isinstance(data1, tuple)
assert target0 < 10
assert (target1 < 20) & (target1 > 10)
assert data0[0].size() == (1, 28, 28)  # the image
assert data0[1].size()[0] == dim_context  # the context
assert sparse_binary_tuple_mnist.dim_context == dim_context

sparse_binary_concat_mnist = ContextDependentPermutedMNIST(
    num_tasks=num_tasks,
    seed=seed,
    train=train,
    download=False,
    dim_context=dim_context,
    combine_context_as="concatenate"
)

data0, target0 = sparse_binary_concat_mnist[0]
data1, target1 = sparse_binary_concat_mnist[65_432]  # second task, first permutation

assert len(sparse_binary_concat_mnist) == 60_000 * num_tasks
assert len(sparse_binary_concat_mnist.data) == 60_000
assert isinstance(data0, torch.Tensor)
assert isinstance(data1, torch.Tensor)
assert target0 < 10
assert (target1 < 20) & (target1 > 10)
assert data0.size()[0] == data1.size()[0] == 784 + dim_context
assert sparse_binary_concat_mnist.dim_context == dim_context


one_hot_tuple_mnist = ContextDependentPermutedMNIST(
    num_tasks=num_tasks,
    seed=seed,
    train=train,
    download=False,
    dim_context=dim_context,
    context_type="one_hot",
    combine_context_as="tuple",
)

data0, target0 = one_hot_tuple_mnist[0]
data1, target1 = one_hot_tuple_mnist[65_432]  # second task, first permutation

assert len(one_hot_tuple_mnist) == 60_000 * num_tasks
assert len(one_hot_tuple_mnist.data) == 60_000
assert isinstance(data0, tuple)
assert isinstance(data1, tuple)
assert target0 < 10
assert (target1 < 20) & (target1 > 10)
assert data0[0].size() == (1, 28, 28)  # the image
assert data0[1].size()[0] == num_tasks  # the context
assert one_hot_tuple_mnist.dim_context == num_tasks


one_hot_concat_mnist = ContextDependentPermutedMNIST(
    num_tasks=num_tasks,
    seed=seed,
    train=train,
    download=False,
    dim_context=dim_context,
    context_type="one_hot",
    combine_context_as="concatenate"
)

data0, target0 = one_hot_concat_mnist[0]
data1, target1 = one_hot_concat_mnist[65_432]  # second task, first permutation

assert len(one_hot_concat_mnist) == 60_000 * num_tasks
assert len(one_hot_concat_mnist.data) == 60_000
assert isinstance(data0, torch.Tensor)
assert isinstance(data1, torch.Tensor)
assert target0 < 10
assert (target1 < 20) & (target1 > 10)
assert data0.size()[0] == data1.size()[0] == 784 + num_tasks
assert one_hot_concat_mnist.dim_context == num_tasks


centroid_tuple_mnist = ContextDependentPermutedMNIST(
    num_tasks=num_tasks,
    seed=seed,
    train=train,
    download=False,
    dim_context=dim_context,
    context_type="centroid",
    combine_context_as="tuple",
)

data0, target0 = centroid_tuple_mnist[0]
data1, target1 = centroid_tuple_mnist[65_432]  # second task, first permutation

assert len(centroid_tuple_mnist) == 60_000 * num_tasks
assert len(centroid_tuple_mnist.data) == 60_000
assert isinstance(data0, tuple)
assert isinstance(data1, tuple)
assert target0 < 10
assert (target1 < 20) & (target1 > 10)
assert data0[0].size() == (1, 28, 28)  # the image
assert data0[1].size()[0] == 784  # the context
assert centroid_tuple_mnist.dim_context == 784

manual_centroids = torch.zeros((num_tasks, 784))
for index in range(len(centroid_tuple_mnist)):
    data, target = centroid_tuple_mnist[index]
    img, context = data
    task_id = index // len(centroid_tuple_mnist.data)
    manual_centroids[task_id] += img.flatten()

manual_centroids /= len(centroid_tuple_mnist.data)

disagreement_no_permutations = torch.abs(
    manual_centroids[0] - centroid_tuple_mnist.contexts[0]).sum()
disagreement_permutation_2 = torch.abs(
    manual_centroids[2] - centroid_tuple_mnist.contexts[2]).sum()

# breaking apart variables to keep line lengths short enoguh for flake8
fmsg = "Absolute difference between centroids task"
print(f"{fmsg} 0 for two methods: mnist raw: {disagreement_no_permutations}")
print(f"{fmsg} 2 for two methods: mnist raw: {disagreement_permutation_2}")

assert disagreement_no_permutations < .001
assert disagreement_permutation_2 < .001


centroid_concat_mnist = ContextDependentPermutedMNIST(
    num_tasks=num_tasks,
    seed=seed,
    train=train,
    download=False,
    dim_context=dim_context,
    context_type="centroid",
    combine_context_as="concatenate"
)

data0, target0 = centroid_concat_mnist[0]
data1, target1 = centroid_concat_mnist[65_432]  # second task, first permutation

assert len(centroid_concat_mnist) == 60_000 * num_tasks
assert len(centroid_concat_mnist.data) == 60_000
assert isinstance(data0, torch.Tensor)
assert isinstance(data1, torch.Tensor)
assert target0 < 10
assert (target1 < 20) & (target1 > 10)
assert data0.size()[0] == data1.size()[0] == 784 + 784
assert centroid_concat_mnist.dim_context == 784
