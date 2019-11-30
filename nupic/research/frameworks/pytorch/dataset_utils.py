# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
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
import collections
import itertools
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader, make_dataset


def create_validation_data_sampler(dataset, ratio):
    """Create `torch.utils.data.Sampler` used to split the dataset into 2
    ramdom sampled subsets. The first should used for training and the second
    for validation.

    :param dataset: A valid torch.utils.data.Dataset (i.e. torchvision.datasets.MNIST)
    :param ratio: The percentage of the dataset to be used for training. The
                  remaining (1-ratio)% will be used for validation
    :return: tuple with 2 torch.utils.data.Sampler. (train, validate)
    """
    indices = np.random.permutation(len(dataset))
    training_count = int(len(indices) * ratio)
    train = torch.utils.data.SubsetRandomSampler(indices=indices[:training_count])
    validate = torch.utils.data.SubsetRandomSampler(indices=indices[training_count:])
    return (train, validate)


class UnionDataset(Dataset):
    """Dataset used to create unions of two or more datasets. The union is
    created by applying the given transformation to the items in the dataset.

    :param datasets: list of datasets of the same size to merge
    :param transform: function used to merge 2 items in the datasets
    """

    def __init__(self, datasets, transform):

        size = len(datasets[0])
        for ds in datasets:
            assert size == len(ds)

        self.datasets = datasets
        self.transform = transform

    def __getitem__(self, index):
        """Return the union value and labels for the item in all datasets.

        :param index: The item to get
        :return: tuple with the merged data and labels associated with the data
        """
        union_data = None
        union_labels = []
        dtype = None
        device = None
        for i, ds in enumerate(self.datasets):
            data, label = ds[index]
            if i == 0:
                union_data = data
                dtype = label.dtype
                device = label.device
            else:
                union_data = self.transform(union_data, data)
            union_labels.append(label)

        return union_data, torch.tensor(union_labels, dtype=dtype, device=device)

    def __len__(self):
        return len(self.datasets[0])


def split_dataset(dataset, groupby):
    """Split the given dataset into multiple datasets grouped by the given
    groupby function. For example::

        # Split mnist dataset into 10 datasets, one dataset for each label
        splitDataset(mnist, groupby=lambda x: x[1])

        # Split mnist dataset into 5 datasets, one dataset for each label pair:
        # [0,1], [2,3],...
        splitDataset(mnist, groupby=lambda x: x[1] // 2)

    :param dataset: Source dataset to split
    :param groupby: Group by function. See :func:`itertools.groupby`
    :return: List of datasets
    """
    # Split dataset based on the group by function and keep track of indices
    indices_by_group = collections.defaultdict(list)
    for k, g in itertools.groupby(enumerate(dataset), key=lambda x: groupby(x[1])):
        indices_by_group[k].extend([i[0] for i in g])

    # Sort by group and create a Subset dataset for each of the group indices
    _, indices = list(
        zip(*(sorted(list(indices_by_group.items()), key=lambda x: x[0])))
    )
    return [Subset(dataset, indices=i) for i in indices]


class PreprocessedDataset(Dataset):
    def __init__(self, cachefilepath, basename, qualifiers):
        """
        A Pytorch Dataset class representing a pre-generated processed dataset stored in
        an efficient compressed numpy format (.npz). The dataset is represented by
        num_files copies, where each copy is a different variation of the full dataset.
        For example, for training with data augmentation, each copy might have been
        generated with a different random seed.  This class is useful if the
        pre-processing time is a significant fraction of training time.

        :param cachefilepath: String for the directory containing pre-processed data.

        :param basename: Base file name from which to construct actual file names.
        Actual file name will be "basename{}.npz".format(i) where i cycles through the
        list of qualifiers.

        :param qualifiers: List of qualifiers for each preprocessed files in this
        dataset.
        """
        self.path = cachefilepath
        self.basename = basename
        self.num_cycle = itertools.cycle(qualifiers)
        self.tensors = []
        self.load_next()

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return len(self.tensors[0])

    def load_next(self):
        """
        Call this to load the next copy into memory, such as at the end of an epoch.

        :return: Name of the file that was actually loaded.
        """
        return self.load_qualifier(next(self.num_cycle))

    def load_qualifier(self, qualifier):
        """
        Call this to load the a copy of a dataset with the specific qualifier into
        memory.

        :return: Name of the file that was actually loaded.
        """
        file_name = os.path.join(self.path, self.basename + "{}.npz".format(qualifier))
        self.tensors = list(np.load(file_name).values())
        return file_name


class CachedDatasetFolder(DatasetFolder):
    """A cached version of `torchvision.datasets.DatasetFolder` where the
    classes and image list are static and cached skiping the costly `os.walk`
    and `os.scandir` calls
    """

    def __init__(
        self,
        root,
        loader=default_loader,
        extensions=IMG_EXTENSIONS,
        transform=None,
        target_transform=None,
        is_valid_file=None,
        num_classes=1000,
    ):
        super(DatasetFolder, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        # Check for cached files
        cache_filename = os.path.join(root, "__cached_dataset_folder__.p")
        if os.path.exists(cache_filename):
            classes, class_to_idx, samples = pickle.load(open(cache_filename, "rb"))
        else:
            # Cache file list
            classes, class_to_idx = self._find_classes(self.root)
            samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
            if len(samples) == 0:
                raise (
                    RuntimeError(
                        "Found 0 files in subfolders of: "
                        + self.root
                        + "\nSupported extensions are: "
                        + ",".join(extensions)
                    )
                )
            pickle.dump(
                (classes, class_to_idx, samples),
                file=open(cache_filename, "wb"),
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        if num_classes < 1000:
            classes, class_to_idx, samples = self.select_subset(
                classes, class_to_idx, samples, num_classes
            )
        assert len(classes) == num_classes
        assert len(class_to_idx) == num_classes

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    @staticmethod
    def select_subset(classes, class_to_idx, samples, num_classes):
        """
        Selects a subset of the classes based on a given number of classes
        Fixed seed ensures the same classes are always chosen, in either train or val
        Example: num_classes=11 will select same classes as num_classes=10 plus 1 extra
        """
        if num_classes > 1000 or num_classes < 1 or type(num_classes) != int:
            raise ValueError("Num_classes has to be an integer between 1 and 1000")
        # sets seed only locally, doesn't interfere with global numpy seed
        selected_idxs = set(
            np.random.RandomState(seed=2019).choice(1000, num_classes, replace=False)
        )

        # filter classes and class to idx
        subset_class_to_idx = {}
        new_idx = 0
        for key, original_idx in class_to_idx.items():
            if original_idx in selected_idxs:
                # renumber the classes from 0
                subset_class_to_idx[key] = new_idx
                new_idx += 1
        subset_classes = [c for c in classes if c in subset_class_to_idx.keys()]

        # select only the relevant samples
        # required since number of samples per class in training set is not uniform
        counter = collections.Counter()
        for j, (_, original_idx) in enumerate(samples, 1):
            counter[original_idx] = j
        subset_samples = []
        for _class in subset_classes:
            original_idx = class_to_idx[_class]
            # select samples, based on original index
            if original_idx == 0:
                class_samples = samples[: counter[original_idx]]
            else:
                class_samples = samples[
                    counter[original_idx - 1] : counter[original_idx]
                ]
            # extend replacing the original index by the new index
            subset_samples.extend(
                [(s[0], subset_class_to_idx[_class]) for s in class_samples]
            )

        return subset_classes, subset_class_to_idx, subset_samples
