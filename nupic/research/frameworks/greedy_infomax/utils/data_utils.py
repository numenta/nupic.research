import torch
from torchvision import transforms
from copy import deepcopy

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
aug = {
    "randcrop": 64,
    "flip": True,
    "bw_mean": [0.4120],
    "bw_std": [0.2570],
}
transform_unsupervised = get_transforms(val=False, aug=aug)
transform_validation = transform_supervised = get_transforms(val=True, aug=aug)

#base_dataset_args = dict(root="~/nta/data/STL10/", download=False)
base_dataset_args = dict(root="~/nta/data/STL10/stl10_binary", download=False)
unsupervised_dataset_args = deepcopy(base_dataset_args)
unsupervised_dataset_args.update(
    dict(transform=transform_unsupervised, split="unlabeled")
)
supervised_dataset_args = deepcopy(base_dataset_args)
supervised_dataset_args.update(
    dict(transform=transform_supervised, split="train")
)
validation_dataset_args = deepcopy(base_dataset_args)
validation_dataset_args.update(
    dict(transform=transform_validation, split="test")
)
STL10_DATASET_ARGS=dict(
        unsupervised=unsupervised_dataset_args,
        supervised=supervised_dataset_args,
        validation=validation_dataset_args,
    )