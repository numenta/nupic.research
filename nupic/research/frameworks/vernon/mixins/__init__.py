# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
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

from .centroid_context import *
from .composite_loss import CompositeLoss
from .configure_optimizer_param_groups import ConfigureOptimizerParamGroups
from .constrain_parameters import ConstrainParameters
from .context_signal import *
from .cutmix import CutMix, CutMixKnowledgeDistillation
from .delay_load_checkpoint import *
from .dendrite_metrics import *
from .ewc import ElasticWeightConsolidation
from .export_model import ExportModel
from .extra_validations_per_epoch import *
from .gradient_metrics import GradientMetrics
from .hidden_activations import *
from .knowledge_distillation import KnowledgeDistillation, KnowledgeDistillationCL
from .legacy_imagenet_config import LegacyImagenetConfig
from .load_preprocessed_data import LoadPreprocessedData
from .log_backprop_structure import LogBackpropStructure
from .log_covariance import LogCovariance
from .log_every_learning_rate import LogEveryLearningRate
from .log_every_loss import LogEveryLoss
from .lr_range_test import LRRangeTest, create_lr_test_experiment
from .maxup import MaxupPerSample, MaxupStandard
from .multi_cycle_lr import MultiCycleLR
from .noise_robustness_test import NoiseRobustnessTest
from .oml import OnlineMetaLearning
from .permuted_mnist_task_indices import *
from .profile import Profile
from .profile_autograd import ProfileAutograd
from .prune_low_magnitude import PruneLowMagnitude
from .prune_low_snr import PruneLowSNRGlobal, PruneLowSNRLayers
from .reduce_lr_after_task import ReduceLRAfterTask
from .regularize_loss import RegularizeLoss
from .report_max_accuracy import ReportMaxAccuracy
from .representation_overlap import *
from .rezero_weights import RezeroWeights
from .save_final_checkpoint import SaveFinalCheckpoint
from .si import *
from .step_based_logging import *
from .torch_profiler import *
from .track_representation_sparsity import *
from .update_boost_strength import UpdateBoostStrength
from .update_dendrite_boost_strength import UpdateDendriteBoostStrength
from .vary_batch_size import VaryBatchSize
