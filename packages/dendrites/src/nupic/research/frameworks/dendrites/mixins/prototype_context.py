# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2022, Numenta, Inc.  Unless you have an agreement
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

import numpy as np
import torch
from scipy.stats import f

from nupic.research.frameworks.dendrites import evaluate_dendrite_model

__all__ = [
    "PrototypeContext",
    "compute_prototype",
    "infer_prototype",
]


class PrototypeContext(metaclass=abc.ABCMeta):
    """
    When training a dendritic network, use the prototype method for computing context
    vectors (that dendrites receive as input). The context vectors can be either
    1) given during training based on task labels, or 2) constructed during a clustering
    method. During inference time, the closest prototype context vector to the input
    sample (in data space) is selected as the corresponding context vector.

    The dict `prototype_context_args` in the experiment config specifies whether the
    context vector should be given during training (based on task labels) or constructed
    using the clustering method.

    About the clustering method: statistical tests determine if the batch of inputs is
    similar to previously-observed inputs. If the batch is similar, an existing context
    vector is used. If the batch isn't similar to any existing context vector, a new
    context vector is instantiated and used. For more details on clustering, see the
    appendix of the following paper:

        https://www.biorxiv.org/content/10.1101/2021.10.25.465651

    Example config (with default values):
    ```
    config = dict(
        prototype_context_args=dict(
            construct=False
        )
    )
    ```
    """

    def setup_experiment(self, config):
        # Since the prototype vector is an element-wise mean of individual data samples,
        # it's necessarily the same dimension as the input
        model_args = config.get("model_args")
        dim_context = model_args.get("dim_context")
        input_size = model_args.get("input_size")

        super().setup_experiment(config)

        prototype_context_args = config.get("prototype_context_args", {})
        self.construct = prototype_context_args.get("construct", False)

        # Tensor for accumulating each task's prototype vector
        self.contexts = torch.zeros((0, self.model.dim_context))
        self.contexts = self.contexts.to(self.device)

        if self.construct:

            # Store "exemplars" for each context vector as a list of Torch Tensors;
            # these are used to perform statistical tests against a new batch of data to
            # determine if that new batch corresponds to the same task
            self.clusters = []

            # `contexts` needs to be a mutable data type in order to be modified by a
            # nested function (below), so it is simply wrapped in a 1-element list
            self.contexts = [self.contexts]

            # This list keeps track of how many exemplars have been used to compute each
            # context vector since 1) we compute a weighted average, and 2) most
            # exemplars are discarded for memory efficiency
            self.contexts_n = []

            # In order to perform statistical variable transformations (below), there
            # are restrictions on the dimensionality of the input, so subindices
            # randomly sample features and discard others
            self.subindices = np.random.choice(range(input_size), size=dim_context,
                                               replace=False)
            self.subindices.sort()

        else:

            # Since the prototype vector is an element-wise mean of individual data
            # samples it's necessarily the same dimension as the input
            assert dim_context == input_size, ("For prototype experiments `dim_context`"
                                               " must match `input_size`")

    def run_task(self):
        self.train_loader.sampler.set_active_tasks(self.current_task)

        if self.construct:

            self.train_context_fn = construct_prototype(self.clusters, self.contexts,
                                                        self.contexts_n, self.subindices
                                                        )

        else:

            # Find a context vector by computing the prototype of all training examples
            self.context_vector = compute_prototype(self.train_loader).to(self.device)
            self.contexts = torch.cat((self.contexts, self.context_vector.unsqueeze(0)))
            self.train_context_fn = train_prototype(self.context_vector)

        return super().run_task()

    def validate(self, loader=None):
        if loader is None:
            loader = self.val_loader

        if self.construct:
            infer_context_fn = infer_prototype(self.contexts[0], self.subindices)
        else:
            infer_context_fn = infer_prototype(self.contexts)

        # TODO: take out constants in the call below
        return evaluate_dendrite_model(model=self.model,
                                       loader=loader,
                                       device=self.device,
                                       criterion=self.error_loss,
                                       share_labels=True, num_labels=10,
                                       infer_context_fn=infer_context_fn)


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
        context = context_vector.repeat((data.size(0), 1))
        return context

    return _train_prototype


def construct_prototype(clusters, contexts, n_samples_per_prototype, subindices,
                        max_samples_per_cluster=256):
    """
    Returns a function that takes a batch of training examples and performs a clustering
    procedure to determine the appropriate context vector. The resulting context vector
    returned by the function is either a) an existing context vector in `contexts` or
    b) simply the prototype of the batch.

    :param clusters: List of Torch Tensors where the item at position i gives the
                     exemplars representing cluster i
    :param contexts: List containing a single Torch Tensor in which row i gives the ith
                     context vector
    :param n_samples_per_context: List of ints where entry i gives the number of
                                  samples used to compute the ith context (i.e.,
                                  `contexts[0][i]`)
    :param subindices: List/Tensor/Array that can index contexts to select subindices;
                       optional
    :param max_samples_per_cluster: Integer giving the maximum number of data samples
                                    per cluster to store for computing statistical tests
    """

    def _construct_prototype(data):

        # The following variables are declared nonlocal since they are modified by this
        # (inner) function
        nonlocal clusters
        nonlocal contexts
        nonlocal n_samples_per_prototype

        data = data[:, subindices]

        # Due to memory constraints, each Tensor in `clusters` will contain a maximum
        # number of individual exemplars which are then used to compute the prototype
        max_samples_per_cluster = 256
        cluster_id = None

        for j in range(len(clusters)):

            # If already clustered, skip
            if cluster_id is not None:
                continue

            if should_cluster(clusters[j], data):
                cluster_id = j

                # As clusters grow, keeping all exemplars (i.e., the data samples that
                # are used to compute prototype) in memory will be problematic; for this
                # reason we only store `max_samples_per_cluster` examples in memory and
                # discard the rest; the following code implements exactly this while
                # ensuring the prototype vector incorporates all observed data samples
                # even if not stored in memory

                # Update prototype via weighted averaging: the two weights are 1) the
                # number of samples that have contributed towards computing the
                # prototype vectory in memory, and 2) the current batch size
                n = n_samples_per_prototype[j]
                n_cluster = clusters[j].size(0)
                n_batch = data.size(0)

                updated_prototype = n * contexts[0][j] + n_batch * data.mean(dim=0)
                updated_prototype /= (n + n_batch)
                contexts[0][j, :] = updated_prototype

                n_samples_per_prototype[j] += n_batch

                # For computation efficiency, drop some samples out of memory

                # Randomly select which examples in memory will be stored, and which
                # ones from the batch will be stored
                p_cluster = n_cluster / (n_cluster + n_batch)
                p_batch = 1.0 - p_cluster

                n_retain = int(max_samples_per_cluster * p_cluster)
                retain_inds = np.random.choice(range(n_cluster), size=n_retain,
                                               replace=False)

                n_new = int(max_samples_per_cluster * p_batch)
                new_inds = np.random.choice(range(n_batch), size=n_new, replace=False)

                clusters[j] = torch.cat((clusters[j][retain_inds],
                                         data[new_inds]))

        if cluster_id is None:

            # No existing cluster is appropriate for the given batch; create new cluster
            clusters.append(data[:max_samples_per_cluster, :])
            contexts[0] = torch.cat((contexts[0], data.mean(dim=0).unsqueeze(0)))
            n_samples_per_prototype.append(data.size(0))

            cluster_id = len(n_samples_per_prototype) - 1

        return contexts[0][cluster_id].repeat((data.size(0), 1))

    return _construct_prototype


def infer_prototype(contexts, subindices=None):
    """
    Returns a function that takes a batch of test examples and returns a 2D array where
    row i gives the the prototype vector closest to the ith test example.
    """

    def _infer_prototype(data):
        if subindices is not None:
            data = data[:, subindices]
        context = torch.cdist(contexts, data)
        context = context.argmin(dim=0)
        context = contexts[context]
        return context

    return _infer_prototype


# ----------------------- Functions for clustering data samples ---------------------- #

def should_cluster(set1, set2, p=0.9):
    """
    Returns True iff the multivariate two-sample test that compares samples from set1
    and set2 suggests that they "belong to the same distribution"; False otherwise.

    :param set1: 2D Torch Tensor
    :param set2: 2D Torch Tensor
    :param p: Statistical significance threshold
    """
    p_value = two_sample_hotelling_statistic(set1, set2)
    return p_value < p


def two_sample_hotelling_statistic(set1, set2):
    """
    Returns a p-value of whether set1 and set2 share the same underlying data-generating
    process. Note that all matrix inversions in the standard formulation are replaced
    with the Moore-Penrose pseudo-inverse. More details are provided here:

        https://en.wikipedia.org/wiki/Hotelling%27s_T-squared_distribution#Two-sample_st
        atistic

    :param set1: 2D Torch Tensor
    :param set2: 2D Torch Tensor
    """

    # NOTE: The operations performs in this function require float64 datatype since
    # numerical values become extremely small. This requires additional memory and
    # slightly slows down the training process.
    set1 = set1.double()
    set2 = set2.double()

    n1 = set1.size(0)
    n2 = set2.size(0)

    mean1 = set1.mean(dim=0)
    mean2 = set2.mean(dim=0)

    # Sample covariance matrices
    cov1 = torch.matmul((set1 - mean1).T, (set1 - mean1))
    cov1 = cov1 / (n1 - 1)

    cov2 = torch.matmul((set2 - mean2).T, set2 - mean2)
    cov2 = cov2 / (n2 - 1)

    # Unbiased pooled covariance matrix
    cov = (n1 - 1) * cov1 + (n2 - 1) * cov2
    cov = cov / (n1 + n2 - 2)

    # T^2 statistic
    t_squared = torch.matmul((mean1 - mean2).unsqueeze(0), torch.pinverse(cov))
    t_squared = torch.matmul(t_squared, mean1 - mean2)
    t_squared = (n1 * n2 / (n1 + n2)) * t_squared

    # Number of features
    p = set1.size(1)
    n = n1 + n2

    # Transform to F variable
    f_statistic = (n - p - 1) / (p * (n - 2)) * t_squared
    f_statistic = f_statistic.cpu().numpy()
    p_value = f.cdf(f_statistic, p, n - p - 1)

    return p_value
