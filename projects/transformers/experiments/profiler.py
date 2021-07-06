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
from copy import deepcopy

import torch

from trainer_mixins.profiler import inject_profiler_mixin

from .ablations import tiny_bert_sparse_100k_onecycle_lr_kd
from .deepspeed import tiny_bert_sparse_100k_onecycle_lr_kd_deepspeed

PROFILER_ARGS = {
    # "with_stack": True,
    # "record_shapes": True,
    "schedule": torch.profiler.schedule(wait=1, warmup=4, active=5)
}
PROFILER_STEPS = 20

# Baseline for comparison
tiny_bert_sparse_100k_onecycle_lr_kd_profiler = deepcopy(
    tiny_bert_sparse_100k_onecycle_lr_kd
)
tiny_bert_sparse_100k_onecycle_lr_kd_profiler.update(
    max_steps=PROFILER_STEPS,
    do_eval=False,
    trainer_class=inject_profiler_mixin(
        tiny_bert_sparse_100k_onecycle_lr_kd["trainer_class"]
    ),
)
tiny_bert_sparse_100k_onecycle_lr_kd_profiler["trainer_mixin_args"].update(
    profiler=PROFILER_ARGS
)

# Deepspeed version
tiny_bert_sparse_100k_onecycle_lr_kd_deepspeed_profiler = deepcopy(
    tiny_bert_sparse_100k_onecycle_lr_kd_deepspeed
)
tiny_bert_sparse_100k_onecycle_lr_kd_deepspeed_profiler.update(
    max_steps=PROFILER_STEPS,
    do_eval=False,
    trainer_class=inject_profiler_mixin(
        tiny_bert_sparse_100k_onecycle_lr_kd_deepspeed["trainer_class"]
    ),
)
tiny_bert_sparse_100k_onecycle_lr_kd_deepspeed_profiler["trainer_mixin_args"].update(
    profiler=PROFILER_ARGS
)

CONFIGS = dict(
    tiny_bert_sparse_100k_onecycle_lr_kd_profiler=tiny_bert_sparse_100k_onecycle_lr_kd_profiler,  # noqa: E501
    tiny_bert_sparse_100k_onecycle_lr_kd_deepspeed_profiler=tiny_bert_sparse_100k_onecycle_lr_kd_deepspeed_profiler,  # noqa: E501
)
