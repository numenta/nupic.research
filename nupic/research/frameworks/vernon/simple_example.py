# check licenses *** needs to match those used by the rest of the team ***
# ?Readme should be in Markdown 
 
# This is a config file that can be used to run a simple model using the Vernon framework

# It will train a multi-layer perceptron on supervised classification of the MNIST data-set

# *** RUNNING THE EXAMPLE ***
# To run this simple example without modifications on your local machine, use the following terminal command

# *** HYPER-PARAMETERS ***
# Vernon allows the user to specify many hyper-parameters, some of which may be unique to e.g. continual or meta-learning settings
# To run a basic example, the minimal parameters that must be provided by the user are given here

# *** DEFAULT VALUES ***
# To see the default parameters used by models, see the model ?classes in ?
# To see the default parameter values used in experiments, see the classes under Handlers.py

# About Vernon
# Framework to enable easy access to mixins (?class-agnostic methods) that researchers can use to explore a) meta-learning
# b) continual learning c) ?sparse networks (weights and activations)
# Named after x Vernon (affectionatly known as 'Mount Vernon' around Numenta), founder of column theory

# Key files
# In addition to Handlers.py and ?.py (model classes) noted above, Run.py and Run?.py are used to accept the user's config file
# and run an experiment 


#Importing a new data-set from torchvision such as CIFAR10 is easy - just make sure you adjust the input_size to match (e.g. 3,32,32)

# Running using vernon's run function; how to specify config (file name vs name of dictionary in the file)

# Now technically working, although getting weird bug where 1-3 epochs results in training, but 4+ epochs results in 0 accuracy on every run
# The network is clearly improving, and after 3 epochs of training does eventually show the performance (specifically, only shows a non-zero accuracy for the last three
# epochs)

# To use a new data-set... (where are these defined)

#To run using Ray, ray_trainable=SupervisedTrainable needs to be specified


import torch
#from torch import nn
from torchvision import datasets, transforms
#import ray
#from nupic.research.frameworks.pytorch.datasets import omniglot, torchvisiondataset
#import logging
from time import time
#import copy
from nupic.research.frameworks.vernon.handlers import SupervisedExperiment
#from nupic.research.frameworks.vernon.run_experiment.run_with_raytune import run_single_instance
from nupic.research.frameworks.pytorch.models.common_models import (
    OmniglotCNN,
    StandardMLP,
)
from nupic.research.frameworks.vernon.run_experiment.trainables import (
    SupervisedTrainable,
)
import numpy as np

mnist_MLP = dict(
    dataset_class=datasets.mnist,
    dataset_args=dict(
        root="~/nta/datasets",
        download=True,
        transform=transforms.ToTensor(),
    ),
    model_class=StandardMLP,
    model_args=dict(
        input_size=(28,28),
        num_classes=10,
    ),
    batch_size=32,
    epochs=2, 
    epochs_to_validate=np.arange(2), #a list of the epochs to evaluate accuracy on; by default only the last three epochs
    num_classes=10,
    distributed=False,
    experiment_class=SupervisedExperiment, #General experiment class used to train neural networks in supervised learning tasks.
    optimizer_class=torch.optim.Adam,
    optimizer_args=dict(lr=1e-4),
)


def run_experiment(config):
    exp = config.get("experiment_class")()
    exp.setup_experiment(config)
    print(f"Training started....")
    while not exp.should_stop():
        t0 = time()
        # print(f"Starting epoch: {exp.get_current_epoch()}")
        result = exp.run_epoch()
        print(f"Finished Epoch: {exp.get_current_epoch()}")
        print(f"Epoch Duration: {time()-t0:.1f}")
        print(f"Accuracy: {result['mean_accuracy']:.4f}")
    print(f"....Training finished")


run_experiment(mnist_MLP)


# # Write a custom run function
# def run_experiment(config):
#     exp = config.get("experiment_class")()
#     exp.setup_experiment(config)
#     while not exp.should_stop():
#         result = exp.run_epoch()

# run_experiment(meta_cl_omniglot)
# # Use one of the existing run functions
# #run_single_instance(meta_cl_omniglot)







# import copy

# import torch
# from torchvision import datasets, transforms

# # from nupic.research.frameworks.pytorch.datasets import omniglot, torchvisiondataset
# # from nupic.research.frameworks.vernon.handlers import (
# #     SupervisedExperiment,
# #     mixins,
# # )
# from nupic.research.frameworks.vernon.handlers import SupervisedExperiment
# from nupic.research.frameworks.vernon.run_experiment.trainables import (
#     SupervisedTrainable,
# )
# from nupic.research.frameworks.pytorch.models import OmniglotCNN, StandardMLP

# from base import DEFAULT


# """
# Base Experiment to test MNIST
# """


# # class ReduceLRContinualLearningExperiment(mixins.ReduceLRAfterTask,
# #                                           ContinualLearningExperiment):
# #     pass


# # baseline - suggested in April 4th as best values possible
# simple_experiment_mnist = copy.deepcopy(DEFAULT)
# simple_experiment_mnist.update(
#     # specific to continual learning
#     distributed=False,
#     ray_trainable=False,
#     experiment_class=SupervisedExperiment,
#     evaluation_metrics=[
#         "eval_current_task",
#         "eval_first_task",
#     ],
#     num_classes=10,
#     # regular experiments
#     dataset_class=datasets.MNIST,
#     model_class=StandardMLP,
#     model_args=dict(
#         input_size=(28, 28),
#         num_classes=10,
#     ),
#     dataset_args=dict(
#         root="~/nta/datasets",
#         transform=transforms.ToTensor(),
#     ),
#     seed=123,
#     # epochs
#     epochs_to_validate=[],
#     epochs=5,
#     # batches_in_epoch=20,
#     batch_size=64,
#     # optimizer
#     optimizer_class=torch.optim.SGD,
#     optimizer_args=dict(
#         lr=0.01,
#         # weight_decay=1e-6,
#         momentum=0.9,
#         nesterov=False,
#     ),
#     # Learning rate scheduler class. Must inherit from "_LRScheduler"
#     lr_scheduler_class=None,
#     lr_scheduler_args=None,
# )

# # cl_mnist_b = copy.deepcopy(cl_mnist)
# # cl_mnist_b.update(
# #     dataset_class=torchvisiondataset,
# #     dataset_args=dict(
# #         root="~/nta/datasets",
# #         dataset_name="MNIST",
# #     ),
# # )


# # cl_omniglot = copy.deepcopy(cl_mnist)
# # cl_omniglot.update(
# #     # logging
# #     # TODO: fix logging on wandb in continuous learning
# #     # wandb_args=dict(
# #     #     name="cl_omniglot",
# #     #     project="test_cl",
# #     #     notes="""
# #     #         Testing CL implementation for Continuous Learning
# #     #     """,
# #     # ),
# #     evaluation_metrics=[
# #         "eval_current_task",
# #         "eval_first_task",
# #         "eval_all_visited_tasks",
# #         "eval_all_tasks",
# #         "eval_individual_tasks"
# #     ],
# #     # dataset specific
# #     dataset_class=omniglot,
# #     num_tasks=2,
# #     num_classes=10,
# #     # define training time
# #     epochs=10,
# #     batch_size=32,
# #     batches_in_epoch=20,
# #     # model
# #     ray_trainable=ContinualLearningTrainable,
# #     model_class=OmniglotCNN,
# #     model_args=dict(
# #         input_size=(105, 105),
# #         num_classes=10,
# #     ),
# #     dataset_args=dict(
# #         root="~/nta/datasets",
# #     ),
# #     optimizer_class=torch.optim.SGD,
# #     optimizer_args=dict(
# #         lr=.1,
# #         momentum=0.9,
# #         weight_decay=1e-8,
# #     ),
# # )

# # Export all configurations
# CONFIGS = dict(
#     simple_experiment_mnist=simple_experiment_mnist
# )
