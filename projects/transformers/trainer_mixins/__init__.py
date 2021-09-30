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

from .deepspeed import DeepspeedTransformerLayerMixin
from .distillation import DistillationTrainerMixin
from .gmp import GradualMagnitudePruningMixin, ThreeStageLRMixin
from .lr_range_test import LRRangeTestMixin
from .multi_eval_sets import MultiEvalSetsTrainerMixin
from .one_cycle_lr import OneCycleLRMixin
from .profiler import TorchProfilerMixin
from .qa_trainer import QuestionAnsweringMixin
from .rigl import RigLMixin
