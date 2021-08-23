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
#
# This work was based on the original Greedy InfoMax codebase from Sindy Lowe:
# https://github.com/loeweX/Greedy_InfoMax
# The Greedy InfoMax paper can be found here:
# https://arxiv.org/abs/1905.11786
# ----------------------------------------------------------------------


from copy import deepcopy

from torchvision import transforms


# get transforms for the dataset
def get_transforms(val=False, aug=None):
    trans = []

    if aug["randcrop"]:
        if not val:
            trans.append(transforms.RandomCrop(aug["randcrop"]))
        else:
            trans.append(transforms.CenterCrop(aug["randcrop"]))

    if aug["flip"] and not val:
        trans.append(transforms.RandomHorizontalFlip())

    trans.append(transforms.Grayscale())
    trans.append(transforms.ToTensor())
    trans.append(transforms.Normalize(mean=aug["bw_mean"], std=aug["bw_std"]))

    trans = transforms.Compose(trans)
    return trans


# labeled train set: mean [0.4469, 0.4400, 0.4069], std [0.2603, 0.2566, 0.2713]
aug = {"randcrop": 64, "flip": True, "bw_mean": [0.4120], "bw_std": [0.2570]}
transform_unsupervised = get_transforms(val=False, aug=aug)
transform_validation = transform_supervised = get_transforms(val=True, aug=aug)

base_dataset_args = dict(root="~/nta/data/STL10/", download=False)
# base_dataset_args = dict(root="~/nta/data/STL10/stl10_binary", download=False)
unsupervised_dataset_args = deepcopy(base_dataset_args)
unsupervised_dataset_args.update(
    dict(transform=transform_unsupervised, split="unlabeled")
)
supervised_dataset_args = deepcopy(base_dataset_args)
supervised_dataset_args.update(dict(transform=transform_supervised, split="train"))
validation_dataset_args = deepcopy(base_dataset_args)
validation_dataset_args.update(dict(transform=transform_validation, split="test"))
STL10_DATASET_ARGS = dict(
    unsupervised=unsupervised_dataset_args,
    supervised=supervised_dataset_args,
    validation=validation_dataset_args,
)

def patchify_inputs(x, patch_size, overlap):
    x = (
        x.unfold(2, patch_size, patch_size // overlap)
        .unfold(3, patch_size, patch_size // overlap)
        .permute(0, 2, 3, 1, 4, 5)  # b, p_x, p_y, c, x, y
    )
    n_patches_x = x.shape[1]
    n_patches_y = x.shape[2]
    x = x.reshape(
        x.shape[0] * x.shape[1] * x.shape[2], x.shape[3], x.shape[4], x.shape[5]
    )
    return x, n_patches_x, n_patches_y
