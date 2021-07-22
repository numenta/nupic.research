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

from .trifecta import tiny_bert_trifecta_100k

"""
These experiments attempt to vary the width and the number of on-params of a BERT model
while maintaining some fixed level of loss. They differ from `wide_bert.py` as that file
only varies the width while keeping the on-params fixed.
"""

# Set training to be distributed.
tiny_bert_trifecta_100k_dist = deepcopy(tiny_bert_trifecta_100k)
tiny_bert_trifecta_100k_dist.update(
    # Using batch_size of 16 instead of 128 since we're training on 8 GPUs.
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
)

# Specify the max lr for 300k on-params.
tiny_bert_trifecta_300k_params = deepcopy(tiny_bert_trifecta_100k_dist)
tiny_bert_trifecta_300k_params["trainer_mixin_args"].update(
    max_lr=0.010,
)
# Specify the max lr for 400k on-params.
tiny_bert_trifecta_400k_params = deepcopy(tiny_bert_trifecta_100k_dist)
tiny_bert_trifecta_400k_params["trainer_mixin_args"].update(
    max_lr=0.009,
)
# Specify the max lr for 500k on-params.
tiny_bert_trifecta_500k_params = deepcopy(tiny_bert_trifecta_100k_dist)
tiny_bert_trifecta_500k_params["trainer_mixin_args"].update(
    max_lr=0.0085,
)
# Specify the max lr for 600k on-params.
tiny_bert_trifecta_600k_params = deepcopy(tiny_bert_trifecta_100k_dist)
tiny_bert_trifecta_600k_params["trainer_mixin_args"].update(
    max_lr=0.0081,
)
# Specify the max lr for 700k on-params.
tiny_bert_trifecta_700k_params = deepcopy(tiny_bert_trifecta_100k_dist)
tiny_bert_trifecta_700k_params["trainer_mixin_args"].update(
    max_lr=0.0078,
)
# Specify the max lr for 800k on-params.
tiny_bert_trifecta_800k_params = deepcopy(tiny_bert_trifecta_100k_dist)
tiny_bert_trifecta_800k_params["trainer_mixin_args"].update(
    max_lr=0.0075,
)
# Specify the max lr for 900k on-params.
tiny_bert_trifecta_900k_params = deepcopy(tiny_bert_trifecta_100k_dist)
tiny_bert_trifecta_900k_params["trainer_mixin_args"].update(
    max_lr=0.0071,
)


# Bert with layers of 4x wider
tiny_bert_4x_wide_args = dict(
    hidden_size=512,
    intermediate_size=2048,
    sparsify_all_embeddings=False,
)


# Bert with layers of 2x wider
tiny_bert_2x_wide_args = dict(
    hidden_size=256,
    intermediate_size=1024,
    sparsify_all_embeddings=False,
)


# --------
# 1x Wide
# --------

# ~300k on-params
tiny_bert_trifecta_1x_wide_9318sparse_100k = deepcopy(tiny_bert_trifecta_300k_params)
tiny_bert_trifecta_1x_wide_9318sparse_100k["config_kwargs"].update(
    sparsity=0.9318,
)

# ~400k on-params
tiny_bert_trifecta_1x_wide_9075sparse_100k = deepcopy(tiny_bert_trifecta_400k_params)
tiny_bert_trifecta_1x_wide_9075sparse_100k["config_kwargs"].update(
    sparsity=0.9075,
)

# ~500k on-params
tiny_bert_trifecta_1x_wide_8831sparse_100k = deepcopy(tiny_bert_trifecta_500k_params)
tiny_bert_trifecta_1x_wide_8831sparse_100k["config_kwargs"].update(
    sparsity=0.8831,
)

# ~600k on-params
tiny_bert_trifecta_1x_wide_8588sparse_100k = deepcopy(tiny_bert_trifecta_600k_params)
tiny_bert_trifecta_1x_wide_8588sparse_100k["config_kwargs"].update(
    sparsity=0.8588,
)

# ~700k on-params
tiny_bert_trifecta_1x_wide_8344sparse_100k = deepcopy(tiny_bert_trifecta_700k_params)
tiny_bert_trifecta_1x_wide_8344sparse_100k["config_kwargs"].update(
    sparsity=0.8344,
)

# ~800k on-params
tiny_bert_trifecta_1x_wide_81sparse_100k = deepcopy(tiny_bert_trifecta_800k_params)
tiny_bert_trifecta_1x_wide_81sparse_100k["config_kwargs"].update(
    sparsity=0.81,
)

# ~900k on-params
tiny_bert_trifecta_1x_wide_7857sparse_100k = deepcopy(tiny_bert_trifecta_900k_params)
tiny_bert_trifecta_1x_wide_7857sparse_100k["config_kwargs"].update(
    sparsity=0.7857,
)


# --------
# 2x Wide
# --------

# ~300k on-params
tiny_bert_trifecta_2x_wide_9711sparse_100k = deepcopy(tiny_bert_trifecta_300k_params)
tiny_bert_trifecta_2x_wide_9711sparse_100k["config_kwargs"].update(
    sparsity=0.9711,
    **tiny_bert_2x_wide_args,
)

# ~400k on-params
tiny_bert_trifecta_2x_wide_96sparse_100k = deepcopy(tiny_bert_trifecta_400k_params)
tiny_bert_trifecta_2x_wide_96sparse_100k["config_kwargs"].update(
    sparsity=0.96,
    **tiny_bert_2x_wide_args,
)

# ~500k on-params
tiny_bert_trifecta_2x_wide_9489sparse_100k = deepcopy(tiny_bert_trifecta_500k_params)
tiny_bert_trifecta_2x_wide_9489sparse_100k["config_kwargs"].update(
    sparsity=0.9489,
    **tiny_bert_2x_wide_args,
)

# ~600k on-params
tiny_bert_trifecta_2x_wide_9378sparse_100k = deepcopy(tiny_bert_trifecta_600k_params)
tiny_bert_trifecta_2x_wide_9378sparse_100k["config_kwargs"].update(
    sparsity=0.9378,
    **tiny_bert_2x_wide_args,
)

# ~700k on-params
tiny_bert_trifecta_2x_wide_9267sparse_100k = deepcopy(tiny_bert_trifecta_700k_params)
tiny_bert_trifecta_2x_wide_9267sparse_100k["config_kwargs"].update(
    sparsity=0.9267,
    **tiny_bert_2x_wide_args,
)

# ~800k on-params
tiny_bert_trifecta_2x_wide_9156sparse_100k = deepcopy(tiny_bert_trifecta_800k_params)
tiny_bert_trifecta_2x_wide_9156sparse_100k["config_kwargs"].update(
    sparsity=0.9156,
    **tiny_bert_2x_wide_args,
)

# ~900k on-params
tiny_bert_trifecta_2x_wide_9045sparse_100k = deepcopy(tiny_bert_trifecta_900k_params)
tiny_bert_trifecta_2x_wide_9045sparse_100k["config_kwargs"].update(
    sparsity=0.9045,
    **tiny_bert_2x_wide_args,
)


# --------
# 4x Wide
# --------

# ~300k on-params
tiny_bert_trifecta_4x_wide_9896sparse_100k = deepcopy(tiny_bert_trifecta_300k_params)
tiny_bert_trifecta_4x_wide_9896sparse_100k["config_kwargs"].update(
    sparsity=0.9896,
    **tiny_bert_4x_wide_args,
)

# ~400k on-params
tiny_bert_trifecta_4x_wide_9849sparse_100k = deepcopy(tiny_bert_trifecta_400k_params)
tiny_bert_trifecta_4x_wide_9849sparse_100k["config_kwargs"].update(
    sparsity=0.9849,
    **tiny_bert_4x_wide_args,
)

# ~500k on-params
tiny_bert_trifecta_4x_wide_9802sparse_100k = deepcopy(tiny_bert_trifecta_500k_params)
tiny_bert_trifecta_4x_wide_9802sparse_100k["config_kwargs"].update(
    sparsity=0.9802,
    **tiny_bert_4x_wide_args,
)

# ~600k on-params
tiny_bert_trifecta_4x_wide_9754sparse_100k = deepcopy(tiny_bert_trifecta_600k_params)
tiny_bert_trifecta_4x_wide_9754sparse_100k["config_kwargs"].update(
    sparsity=0.9754,
    **tiny_bert_4x_wide_args,
)

# ~700k on-params
tiny_bert_trifecta_4x_wide_9707sparse_100k = deepcopy(tiny_bert_trifecta_700k_params)
tiny_bert_trifecta_4x_wide_9707sparse_100k["config_kwargs"].update(
    sparsity=0.9707,
    **tiny_bert_4x_wide_args,
)

# ~800k on-params
tiny_bert_trifecta_4x_wide_966sparse_100k = deepcopy(tiny_bert_trifecta_800k_params)
tiny_bert_trifecta_4x_wide_966sparse_100k["config_kwargs"].update(
    sparsity=0.966,
    **tiny_bert_4x_wide_args,
)

# ~900k on-params
tiny_bert_trifecta_4x_wide_9612sparse_100k = deepcopy(tiny_bert_trifecta_900k_params)
tiny_bert_trifecta_4x_wide_9612sparse_100k["config_kwargs"].update(
    sparsity=0.9612,
    **tiny_bert_4x_wide_args,
)


CONFIGS = dict(
    # 1x wide
    tiny_bert_trifecta_1x_wide_9318sparse_100k=tiny_bert_trifecta_1x_wide_9318sparse_100k,  # noqa: E501
    tiny_bert_trifecta_1x_wide_9075sparse_100k=tiny_bert_trifecta_1x_wide_9075sparse_100k,  # noqa: E501
    tiny_bert_trifecta_1x_wide_8831sparse_100k=tiny_bert_trifecta_1x_wide_8831sparse_100k,  # noqa: E501
    tiny_bert_trifecta_1x_wide_8588sparse_100k=tiny_bert_trifecta_1x_wide_8588sparse_100k,  # noqa: E501
    tiny_bert_trifecta_1x_wide_8344sparse_100k=tiny_bert_trifecta_1x_wide_8344sparse_100k,  # noqa: E501
    tiny_bert_trifecta_1x_wide_81sparse_100k=tiny_bert_trifecta_1x_wide_81sparse_100k,  # noqa: E501
    tiny_bert_trifecta_1x_wide_7857sparse_100k=tiny_bert_trifecta_1x_wide_7857sparse_100k,  # noqa: E501
    # 2x wide
    tiny_bert_trifecta_2x_wide_9711sparse_100k=tiny_bert_trifecta_2x_wide_9711sparse_100k,  # noqa: E501
    tiny_bert_trifecta_2x_wide_96sparse_100k=tiny_bert_trifecta_2x_wide_96sparse_100k,  # noqa: E501
    tiny_bert_trifecta_2x_wide_9489sparse_100k=tiny_bert_trifecta_2x_wide_9489sparse_100k,  # noqa: E501
    tiny_bert_trifecta_2x_wide_9378sparse_100k=tiny_bert_trifecta_2x_wide_9378sparse_100k,  # noqa: E501
    tiny_bert_trifecta_2x_wide_9267sparse_100k=tiny_bert_trifecta_2x_wide_9267sparse_100k,  # noqa: E501
    tiny_bert_trifecta_2x_wide_9156sparse_100k=tiny_bert_trifecta_2x_wide_9156sparse_100k,  # noqa: E501
    tiny_bert_trifecta_2x_wide_9045sparse_100k=tiny_bert_trifecta_2x_wide_9045sparse_100k,  # noqa: E501
    # 4x wide
    tiny_bert_trifecta_4x_wide_9896sparse_100k=tiny_bert_trifecta_4x_wide_9896sparse_100k,  # noqa: E501
    tiny_bert_trifecta_4x_wide_9849sparse_100k=tiny_bert_trifecta_4x_wide_9849sparse_100k,  # noqa: E501
    tiny_bert_trifecta_4x_wide_9802sparse_100k=tiny_bert_trifecta_4x_wide_9802sparse_100k,  # noqa: E501
    tiny_bert_trifecta_4x_wide_9754sparse_100k=tiny_bert_trifecta_4x_wide_9754sparse_100k,  # noqa: E501
    tiny_bert_trifecta_4x_wide_9707sparse_100k=tiny_bert_trifecta_4x_wide_9707sparse_100k,  # noqa: E501
    tiny_bert_trifecta_4x_wide_966sparse_100k=tiny_bert_trifecta_4x_wide_966sparse_100k,  # noqa: E501
    tiny_bert_trifecta_4x_wide_9612sparse_100k=tiny_bert_trifecta_4x_wide_9612sparse_100k,  # noqa: E501
)
