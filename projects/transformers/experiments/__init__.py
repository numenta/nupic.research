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

# Automatically import models. This will update Transformer's model mappings so that
# custom models can be loaded via AutoModelForMaskedLM and related auto-constructors.
import models

from .ablations import CONFIGS as ABLATIONS
from .base import CONFIGS as BASE
from .bert_replication import CONFIGS as BERT_REPLICATION
from .bertitos import CONFIGS as BERTITOS
from .deepspeed import CONFIGS as DEEPSPEED
from .distillation import CONFIGS as DISTILLATION
from .eighty_percent_sparse import CONFIGS as EIGHT_PERCENT_SPARSE
from .finetuning import CONFIGS as FINETUNING
from .hpchase import CONFIGS as HPCHASE
from .hpsearch import CONFIGS as HPSEARCH
from .gmp_bert import CONFIGS as GMP_BERT
from .one_cycle_lr import CONFIGS as ONE_CYCLE_LR
from .profiler import CONFIGS as PROFILER
from .regressions import CONFIGS as REGRESSIONS
from .rigl_bert import CONFIGS as RIGL_BERT
from .sparse_bert import CONFIGS as SPARSE_BERT
from .sparse_bertitos import CONFIGS as SPARSE_BERTITOS
from .trifecta import CONFIGS as TRIFECTA
from .wide_bert import CONFIGS as WIDE_BERT
from .wide_bert_fixed_num_params import CONFIGS as WIDE_BERT_FIXED_NUM_PARAMS

"""
Import and collect all experiment configurations into one CONFIG
"""
__all__ = ["CONFIGS"]

# Collect all configurations
CONFIGS = dict()
CONFIGS.update(ABLATIONS)
CONFIGS.update(BASE)
CONFIGS.update(BERT_REPLICATION)
CONFIGS.update(BERTITOS)
CONFIGS.update(DEEPSPEED)
CONFIGS.update(DISTILLATION)
CONFIGS.update(EIGHT_PERCENT_SPARSE)
CONFIGS.update(FINETUNING)
CONFIGS.update(HPSEARCH)
CONFIGS.update(HPCHASE)
CONFIGS.update(GMP_BERT)
CONFIGS.update(ONE_CYCLE_LR)
CONFIGS.update(PROFILER)
CONFIGS.update(REGRESSIONS)
CONFIGS.update(RIGL_BERT)
CONFIGS.update(SPARSE_BERT)
CONFIGS.update(SPARSE_BERTITOS)
CONFIGS.update(TRIFECTA)
CONFIGS.update(WIDE_BERT)
CONFIGS.update(WIDE_BERT_FIXED_NUM_PARAMS)
