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
from sklearn.metrics import pairwise_distances


def dot_product_metric(x, y):
    return x.dot(y)


class SpatialPoolerAnalysis(metaclass=abc.ABCMeta):
    """
    Analyze the representations coming out of an untrained SP as possible context
    vectors.
    """

    def setup_experiment(self, config):
        model_args = config.get("model_args")
        self.dim_context = model_args.get("output_size")

        super().setup_experiment(config)

        # We're not going to train anything, but need this for k-winner duty
        # cycle
        self.model.train()

        # Tensor for accumulating each task's prototype vector
        self.contexts = torch.zeros((0, self.dim_context))
        self.tasks = []

        # We allow any metric specified in sklearn.metrics.pairwise_distances, plus
        # dot product.
        self.distance_metric = config.get("distance_metric", "dot")

    def train_epoch(self):
        """Don't train anything"""
        pass

    def should_stop(self):
        """Stop after the first task."""
        return self.current_task > 0

    def validate(self, loader=None):
        if loader is None:
            loader = self.val_loader

        contexts = torch.zeros((0, self.dim_context))
        tasks = []

        for task in range(self.num_tasks):
            num_samples = 0
            loader.sampler.set_active_tasks(task)

            with torch.no_grad():
                for data, _ in loader:
                    if isinstance(data, list):
                        data, context = data
                    data = data.flatten(start_dim=1)

                    data = data.to(self.device)
                    output = self.model(data)
                    self.optimizer.zero_grad()
                    num_samples += len(data)
                    contexts = torch.cat((contexts, output))
                    tasks.extend([task] * len(data))

                    if num_samples >= 1000:
                        break

        self.tasks = np.array(tasks)
        self.contexts = contexts.numpy()
        print("Numpy contexts, tasks:",
              self.contexts.shape, self.tasks.shape)
        print("Duty cycle mean/min/max: ",
              self.model.kw.duty_cycle.mean(), self.model.kw.duty_cycle.min(),
              self.model.kw.duty_cycle.max())
        separation = self.compute_distances()
        entropy = float(self.model.kw.entropy())
        return dict(entropy=entropy, mean_accuracy=separation)

    def compute_distances(self):
        """
        Compute the within-task distances and the across task distances of the
        SP outputs. The method returns the 'separation', defined as the ratio
        between the mean inter-class and intra-class distances. The higher this
        number, the more separated the context vectors are.
        """
        metric = self.distance_metric
        if self.distance_metric == "dot":
            metric = dot_product_metric

        avg_dist = np.zeros((self.num_tasks, self.num_tasks))
        stdev_dist = np.zeros((self.num_tasks, self.num_tasks))
        for i in range(self.num_tasks):
            for j in range(self.num_tasks):
                distances = pairwise_distances(
                    self.contexts[self.tasks == i], self.contexts[self.tasks == j],
                    metric=metric)
                avg_dist[i, j] = distances.mean()
                stdev_dist[i, j] = distances.std()

        # avg_dist /= avg_dist.max()
        print("Distance matrix using metric:", self.distance_metric)
        print(avg_dist)
        diag_sum = np.trace(avg_dist)
        diag_mean = diag_sum / avg_dist.shape[0]
        num_off_diag_elements = avg_dist.size - avg_dist.shape[0]
        off_diag_mean = (np.sum(avg_dist) - diag_sum) / num_off_diag_elements

        if self.distance_metric == "dot":
            # Want intra-class dot products to be higher than inter-class dot products
            separation = diag_mean / off_diag_mean
        else:
            separation = off_diag_mean / diag_mean

        print("Distances:", diag_mean, off_diag_mean, separation, self.contexts.size)
        return separation
