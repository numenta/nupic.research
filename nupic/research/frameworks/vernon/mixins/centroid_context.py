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

import numpy as np
import torch
from scipy.stats import f

from nupic.research.frameworks.dendrites import evaluate_dendrite_model

__all__ = [
    "CentroidContext",
    "cluster_centroid_while_training",
    "compute_centroid",
    "infer_centroid",
    "provide_centroid_while_training",
    "should_cluster",
    "two_sample_hotelling_statistic"
]


class CentroidContext(metaclass=abc.ABCMeta):
    """
    When training a dendritic network, use the centroid method for computing context
    vectors (that dendrites receive as input) at either 1) just inference, or 2) both
    training and inference. If computing cetroids while training, a clustering
    algorithm is used which performs statistical tests betweens an existing centroid's
    exemplars and the new batch of data to determine whether said centroid should
    represent said batch.

    The dict `centroid_context_args` in the experiment config specifies whether the
    context vector should be inferred during training. The example below gives default
    values.

    Example config:
    ```
    config = dict(
        centroid_context_args=dict(
            infer_while_training=False
        )
    )
    ```
    """

    def setup_experiment(self, config):
        # Since the centroid vector is an element-wise mean of individual data samples,
        # it's necessarily the same dimension as the input
        model_args = config.get("model_args")
        dim_context = model_args.get("dim_context")
        input_size = model_args.get("input_size")

        super().setup_experiment(config)

        centroid_context_args = config.get("centroid_context_args", {})
        self.infer_while_training = centroid_context_args.get("infer_while_training",
                                                              False)
        self.subindices = None

        if self.infer_while_training:

            # Track clusters of data samples believed to be form the same task
            self.clusters = []

            # `contexts` is a list containing a single tensor since we require a
            # mutable object
            self.contexts = [torch.zeros((0, dim_context))]
            self.n_samples_per_context = []

            # Subsample features from input to form the context; we do this
            # dimensionality reduction because in order to transform a T-squared
            # variable to an F-distributed variable, the dimensionality of the context
            # must be strictly less than the total number of data samples
            self.subindices = np.random.choice(range(input_size), size=dim_context,
                                               replace=False)
            self.subindices.sort()

        else:

            assert dim_context == input_size, ("If centroids are only inferred at "
                                               "inference time, `dim_context` must "
                                               "match `input_size`")

            self.contexts = [torch.zeros((0, input_size))]

        self.contexts[0] = self.contexts[0].to(self.device)

    def run_task(self):
        self.train_loader.sampler.set_active_tasks(self.current_task)

        if self.infer_while_training:
            self.train_context_fn = cluster_centroid_while_training(
                self.clusters, self.contexts, self.n_samples_per_context,
                self.subindices
            )

        else:
            # Construct a context vector by computing the centroid of all training
            # examples
            self.context_vector = compute_centroid(self.train_loader).to(self.device)
            self.contexts[0] = torch.cat((self.contexts[0],
                                          self.context_vector.unsqueeze(0)))
            self.train_context_fn = provide_centroid_while_training(
                self.context_vector
            )

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
                                       infer_context_fn=infer_centroid(
                                           self.contexts, self.subindices)
                                       )


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


def provide_centroid_while_training(context_vector):
    """
    Returns a function that takes a batch of training examples and returns the same
    context vector for each, and that context vector is provided while training, not
    inferred.
    """

    def _train_centroid(data):
        context = context_vector.repeat(data.shape[0], 1)
        return context

    return _train_centroid


def cluster_centroid_while_training(clusters, contexts, n_samples_per_context,
                                    subindices=None):
    """
    Returns a function that takes a batch of training examples and returns a function
    that performs a clustering procedure to determine the appropriate context vector.
    The resulting context vector returned by the function is either a) an existing
    context vector in `contexts` or b) simply the centroid of the batch.

    :param clusters: List of Torch Tensors where the item at position i gives the
                     exemplars representing cluster i
    :param contexts: List containing a single Torch Tensor in which row i gives the ith
                     context vector
    :param n_samples_per_context: List of ints where entry i gives the number of
                                  samples used to compute the ith context (i.e.,
                                  `contexts[0][i]`)
    :param subindices: List/Tensor/Array that can index contexts to select subindices;
                       optional
    """

    def _cluster(data):
        nonlocal contexts
        nonlocal clusters
        nonlocal n_samples_per_context

        # Due to memory constraints, each Tensor in `clusters` will contain a maximum
        # number of individual exemplars which are then used to compute the centroid
        max_samples_per_cluster = 256

        found_cluster = False
        context_id = None

        if subindices is not None:
            data = data[:, subindices]

        for cluster_id in range(len(clusters)):

            # If a suitable cluster has been found, skip ahead
            if found_cluster:
                continue

            if should_cluster(clusters[cluster_id], data):

                found_cluster = True
                context_id = cluster_id

                # As clusters grow, keeping all exemplars (i.e., the data samples that
                # are used to compute centroid) in memory will be problematic; for this
                # reason we only store `max_samples_per_cluster` examples in memory and
                # discard the rest; the following code implements exactly this while
                # ensuring the centroid incorporates all observed data samples even if
                # not stored in memory

                # Update centroid via weighted averaging: the two weights are 1) the
                # number of samples that have contributed towards computing the
                # centroid in memory, and 2) the current batch size
                n = n_samples_per_context[cluster_id]
                n_cluster = clusters[cluster_id].size(0)
                n_batch = data.size(0)

                updated_context = n * contexts[0][cluster_id]\
                    + n_batch * data.mean(dim=0)
                updated_context /= (n + n_batch)
                contexts[0][cluster_id, :] = updated_context

                n_samples_per_context[cluster_id] += n_batch

                p_cluster = n_cluster / (n_cluster + n_batch)
                p_batch = 1.0 - p_cluster

                # Randomly select which examples in memory will be stored, and which
                # ones from the batch will be stored
                num_retain = int(max_samples_per_cluster * p_cluster)
                retain_inds = np.random.choice(range(n_cluster), size=num_retain,
                                               replace=False)

                num_new = int(max_samples_per_cluster * p_batch)
                new_inds = np.random.choice(range(n_batch), size=num_new,
                                            replace=False)

                clusters[cluster_id] = torch.cat((clusters[cluster_id][retain_inds],
                                                  data[new_inds]))

        if not found_cluster:

            # No existing cluster suits the given batch; create a new cluster
            clusters.append(data[:max_samples_per_cluster, :])
            contexts[0] = torch.cat((contexts[0], data.mean(dim=0).unsqueeze(0)))
            n_samples_per_context.append(data.size(0))

            context_id = len(n_samples_per_context) - 1

        context = contexts[0][context_id].repeat(data.shape[0], 1)
        return context

    return _cluster


def infer_centroid(contexts, subindices=None):
    """
    Returns a function that takes a batch of test examples and returns a 2D array where
    row i gives the the centroid vector closest to the ith test example.
    """

    def _infer_centroid(data):
        if subindices is not None:
            data = data[:, subindices]
        context = torch.cdist(contexts[0], data)
        context = context.argmin(dim=0)
        context = contexts[0][context]
        return context

    return _infer_centroid


def should_cluster(set1, set2, p=0.9):
    """
    Returns True iff the multivariate two-sample test that compares samples from set1
    and set2 suggests that they 'belong to the same distribution'; False otherwise

    :param set1: 2D Torch Tensor
    :param set2: 2D Torch Tensor
    :param p: Statistical significance threshold
    """
    p_value = two_sample_hotelling_statistic(set1, set2)
    return p_value < p


def two_sample_hotelling_statistic(set1, set2):
    """
    Returns a p-value of whether set1 and set2 share the same underlying data-
    generating process. Note that all matrix inversions in the standard formulation are
    replaced with the Moore-Penrose pseudo-inverse. More details are provided here:

        https://en.wikipedia.org/wiki/Hotelling%27s_T-squared_distribution#Two-sample_s
        tatistic

    :param set1: 2D Torch Tensor
    :param set2: 2D Torch Tensor
    """
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

    # t-squared statistic
    t_squared = torch.matmul((mean1 - mean2).unsqueeze(0), torch.pinverse(cov))
    t_squared = torch.matmul(t_squared, mean1 - mean2)
    t_squared = (n1 * n2 / (n1 + n2)) * t_squared

    # Number of features
    p = set1.size(1)
    n = n1 + n2

    # Transform t-squared statistic to F-distributed variable
    f_statistic = (n - p - 1) / (p * (n - 2)) * t_squared
    f_statistic = f_statistic.cpu().numpy()
    p_value = f.cdf(f_statistic, p, n - p - 1)

    return p_value
