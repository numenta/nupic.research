# check licenses
import copy

import torch
from torchvision import datasets, transforms

# from nupic.research.frameworks.pytorch.datasets import omniglot, torchvisiondataset
# from nupic.research.frameworks.vernon.handlers import (
#     SupervisedExperiment,
#     mixins,
# )
from nupic.research.frameworks.vernon.handlers import SupervisedExperiment
from nupic.research.frameworks.vernon.run_experiment.trainables import (
    SupervisedTrainable,
)
from nupic.research.frameworks.pytorch.models import OmniglotCNN, StandardMLP

from base import DEFAULT


"""
Base Experiment to test MNIST
"""


# class ReduceLRContinualLearningExperiment(mixins.ReduceLRAfterTask,
#                                           ContinualLearningExperiment):
#     pass


# baseline - suggested in April 4th as best values possible
simple_experiment_mnist = copy.deepcopy(DEFAULT)
simple_experiment_mnist.update(
    # specific to continual learning
    distributed=False,
    ray_trainable=False,
    experiment_class=SupervisedExperiment,
    evaluation_metrics=[
        "eval_current_task",
        "eval_first_task",
    ],
    num_classes=10,
    # regular experiments
    dataset_class=datasets.MNIST,
    model_class=StandardMLP,
    model_args=dict(
        input_size=(28, 28),
        num_classes=10,
    ),
    dataset_args=dict(
        root="~/nta/datasets",
        transform=transforms.ToTensor(),
    ),
    seed=123,
    # epochs
    epochs_to_validate=[],
    epochs=5,
    # batches_in_epoch=20,
    batch_size=64,
    # optimizer
    optimizer_class=torch.optim.SGD,
    optimizer_args=dict(
        lr=0.01,
        # weight_decay=1e-6,
        momentum=0.9,
        nesterov=False,
    ),
    # Learning rate scheduler class. Must inherit from "_LRScheduler"
    lr_scheduler_class=None,
    lr_scheduler_args=None,
)

# cl_mnist_b = copy.deepcopy(cl_mnist)
# cl_mnist_b.update(
#     dataset_class=torchvisiondataset,
#     dataset_args=dict(
#         root="~/nta/datasets",
#         dataset_name="MNIST",
#     ),
# )


# cl_omniglot = copy.deepcopy(cl_mnist)
# cl_omniglot.update(
#     # logging
#     # TODO: fix logging on wandb in continuous learning
#     # wandb_args=dict(
#     #     name="cl_omniglot",
#     #     project="test_cl",
#     #     notes="""
#     #         Testing CL implementation for Continuous Learning
#     #     """,
#     # ),
#     evaluation_metrics=[
#         "eval_current_task",
#         "eval_first_task",
#         "eval_all_visited_tasks",
#         "eval_all_tasks",
#         "eval_individual_tasks"
#     ],
#     # dataset specific
#     dataset_class=omniglot,
#     num_tasks=2,
#     num_classes=10,
#     # define training time
#     epochs=10,
#     batch_size=32,
#     batches_in_epoch=20,
#     # model
#     ray_trainable=ContinualLearningTrainable,
#     model_class=OmniglotCNN,
#     model_args=dict(
#         input_size=(105, 105),
#         num_classes=10,
#     ),
#     dataset_args=dict(
#         root="~/nta/datasets",
#     ),
#     optimizer_class=torch.optim.SGD,
#     optimizer_args=dict(
#         lr=.1,
#         momentum=0.9,
#         weight_decay=1e-8,
#     ),
# )

# Export all configurations
CONFIGS = dict(
    simple_experiment_mnist=simple_experiment_mnist
)
