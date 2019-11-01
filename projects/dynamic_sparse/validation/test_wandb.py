# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
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

# import os

# from dotenv import load_dotenv
# from ray.tune.logger import DEFAULT_LOGGERS

# from nupic.research.frameworks.dynamic_sparse.common.utils import run_ray
# from wandb.ray import WandbLogger


# # alternative initialization based on configuration
# exp_config = dict(
#     device="cpu",
#     network="MLP",
#     dataset_name="MNIST",
#     input_size=784,
#     hidden_sizes=[50,50,50],
#     num_classes=10,
#     model="BaseModel",
#     data_dir="~/nta/datasets",
#     monitor=True,
#     env_config={"wandb": {
#         "project": "test-wandb-project",
#         "sync_tensorboard": True
#     }}
# )

# # run
# tune_config = dict(
#     name="test",
#     num_samples=3,
#     local_dir=os.path.expanduser("~/nta/results"),
#     checkpoint_freq=0,
#     checkpoint_at_end=False,
#     stop={"training_iteration": 30},
#     resources_per_trial={"cpu": 4, "gpu": 0},
#     verbose=2,
#     # loggers=DEFAULT_LOGGERS + (WandbLogger,),
#     loggers = [WandbLogger],
# )

# run_ray(tune_config, exp_config)
