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

from random import uniform

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import Dataset

from nupic.research.frameworks.dendrites.routing import generate_context_vectors


# ------ Dataset
class GaussianDataset(Dataset):
    """
    A dataset class where samples from each category (i.e., class) are drawn i.i.d.
    from a multivariate normal distribution. The mean of each distribution is randomly
    chosen, and its covariance matrix is simply the identity matrix scaled by a random
    scalar value. Each task is also associated with a binary sparse context vector,
    hence the dataset exists in (data, context, target) triplets.
    """

    # Distributions should be the same whether and instance of this class is
    # instantiated for training or testing
    means = None
    covs = None
    _contexts = None

    def __init__(self, num_classes, num_tasks, training_examples_per_class,
                 validation_examples_per_class, dim_x, dim_context,
                 root=None, dataset_name=None, train=True):

        self.num_classes = num_classes
        self.num_tasks = num_tasks
        if train:
            examples_per_class = training_examples_per_class
        else:
            examples_per_class = validation_examples_per_class

        # Initialize disitributions only if `GaussianDataset` object does not exist in
        # memory
        if GaussianDataset.means is None:

            assert GaussianDataset.covs is None
            assert GaussianDataset._contexts is None

            GaussianDataset.means = {class_id: torch.rand((dim_x,)) for class_id in
                                     range(self.num_classes)}
            GaussianDataset.covs = {class_id: uniform(0.1, 2.5) * torch.eye(dim_x) for
                                    class_id in range(self.num_classes)}

        self.distributions = {class_id: MultivariateNormal(
            loc=GaussianDataset.means[class_id],
            covariance_matrix=GaussianDataset.covs[class_id]
        ) for class_id in range(self.num_classes)}

        # Sample i.i.d. from each distribution
        self.data = {}
        for class_id in range(self.num_classes):
            self.data[class_id] = self.distributions[class_id].sample(
                sample_shape=torch.Size([examples_per_class])
            )
        self.data = torch.cat([self.data[class_id] for class_id in
                               range(self.num_classes)], dim=0)

        self.targets = torch.tensor([[class_id for n in range(examples_per_class)]
                                     for class_id in range(self.num_classes)])
        self.targets = self.targets.flatten()

        # Context vectors
        if GaussianDataset._contexts is None:
            GaussianDataset._contexts = generate_context_vectors(num_contexts=num_tasks,
                                                                 n_dim=dim_context,
                                                                 percent_on=0.05)

        num_repeats = int(num_classes * examples_per_class / num_tasks)
        self.contexts = torch.repeat_interleave(GaussianDataset._contexts,
                                                repeats=num_repeats, dim=0)

    def __getitem__(self, idx):
        return self.data[idx, :], self.contexts[idx, :], self.targets[idx].item()

    def __len__(self):
        return self.data.size(0)
