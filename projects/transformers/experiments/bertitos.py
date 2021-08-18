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

"""
These configs serve to provide a smaller version of Bert for rapid prototyping.
"""

from copy import deepcopy

from .base import debug_bert
from .bert_replication import bert_100k

"""
Reduced sized versions of Bert based off the repo:
https://github.com/google-research/bert (See this link for GLUE scores.)

The batch size of 128 was maximized under the constraint that training fit on
a single p3.2xlarge GPU.

Both the train and validation sets are comprised of 1% Wikipedia and 8% of Book Corpus
for a total of 1,132,942 text samples once tokenized with a max_seq_length of 128.
This is roughly 8.7 gigabytes of tokenized data total.
See ../notebooks/create_little_text_dataset.ipynb
"""

#
# Params and Results:
#
# |--------------------------------------------------------------------------|
# | model           | num params | train loss  | eval loss      | perplexity |
# |-----------------|:----------:|:-----------:|:--------------:|:----------:|
# | small_bert_100k | 27,523,072 | 2.951       | 2.676          | 14.532     |
# | small_bert_50k  | 27,523,072 | 3.373       | 3.063          | 21.386     |
# | mini_bert_100k  | 10,615,808 | 3.669       | 3.308          | 27.342     |
# | mini_bert_50k   | 10,615,808 | 4.464       | 4.031          | 56.309     |
# | tiny_bert_100k  |  4,124,928 | 4.566       | 4.021          | 55.793     |
# | tiny_bert_50k   |  4,124,928 | 5.990       | 5.800          | 330.234    |
# |--------------------------------------------------------------------------|
#
# Mixed precision training:
#
# |------------------------------------------------------------------|
# | model                | train loss  | eval loss      | perplexity |
# |----------------------|:-----------:|:--------------:|:----------:|
# | small_bert_100k_fp16 | 2.964       | 2.698          | 14.844     |
# | small_bert_50k_fp16  | 3.286       | 2.979          | 19.671     |
# |------------------------------------------------------------------|
#
# Model Structure
# |--------------------------------------------------------------------------------|
# | model      | attention_heads | hidden_layers | hidden_size | intermediate_size |
# |------------|:---------------:|:-------------:|:-----------:|:-----------------:|
# | small_bert | 8               | 4             | 512         | 2048              |
# | mini_bert  | 4               | 4             | 256         | 1024              |
# | tiny_bert  | 2               | 2             | 128         | 512               |
# |--------------------------------------------------------------------------------|
#
# Training Times:
#
# |--------------------------------------|
# | model                | training time |
# |----------------------|:-------------:|
# | small_bert_100k      | ~8 hrs        |
# | small_bert_100k_fp16 | ~5 hrs 45 min |
# | mini_bert_100k       | ~6 hrs        |
# | tiny_bert_100k       | ~2 hrs 20 min |
# |--------------------------------------|
# (reported using a one p3.2xlarge)
#


# samples/second = 2.536 (on one p3.2xlarge)
small_bert_debug = deepcopy(debug_bert)
small_bert_debug.update(
    # Model Arguments
    model_type="bert",
    config_kwargs=dict(
        num_attention_heads=8,
        num_hidden_layers=4,
        hidden_size=512,  # hidden_size = 64 * num_attention_heads
        intermediate_size=2048,  # intermediate_size = 4 * hidden_size
        max_position_embeddings=128,
    ),
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    max_seq_length=128,
)


# samples/second = 3.989 (on one p3.2xlarge)
mini_bert_debug = deepcopy(debug_bert)
mini_bert_debug.update(
    # Model Arguments
    model_type="bert",
    config_kwargs=dict(
        num_attention_heads=4,
        num_hidden_layers=4,
        hidden_size=256,  # hidden_size = 64 * num_attention_heads
        intermediate_size=1024,  # intermediate_size = 4 * hidden_size
        max_position_embeddings=128,
    ),
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    max_seq_length=128,
)


# samples/second = 5.428 (on one p3.2xlarge)
tiny_bert_debug = deepcopy(debug_bert)
tiny_bert_debug.update(
    # Model Arguments
    model_type="bert",
    config_kwargs=dict(
        num_attention_heads=2,
        num_hidden_layers=2,
        hidden_size=128,  # hidden_size = 64 * num_attention_heads
        intermediate_size=512,  # intermediate_size = 4 * hidden_size
        max_position_embeddings=128,
    ),
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    max_seq_length=128,
)


# samples/second = 2.536 (on one p3.2xlarge)
# This is an even simpler debugging model. Just something to make sure the code can run.
nano_bert_debug = deepcopy(debug_bert)
nano_bert_debug.update(
    # Model Arguments
    model_type="bert",
    config_kwargs=dict(
        num_attention_heads=1,
        num_hidden_layers=1,
        hidden_size=32,  # hidden_size = 64 * num_attention_heads
        intermediate_size=128,  # intermediate_size = 4 * hidden_size
        max_position_embeddings=128,
    ),
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    max_seq_length=128,
    do_eval=False,
)


# samples/second = 3.48 (on one p3.2xlarge)
small_bert_100k = deepcopy(bert_100k)
small_bert_100k.update(
    # Model Arguments
    model_type="bert",
    config_kwargs=dict(
        num_attention_heads=8,
        num_hidden_layers=4,
        hidden_size=512,  # hidden_size = 64 * num_attention_heads
        intermediate_size=2048,  # intermediate_size = 4 * hidden_size
        max_position_embeddings=128,
    ),
    # Training
    per_device_train_batch_size=128,
    per_evice_eval_batch_size=128,
    # Dataset
    max_seq_length=128,
    dataset_name="wikipedia_plus_bookcorpus",
    dataset_config_name=None,
)


small_bert_300k = deepcopy(small_bert_100k)
small_bert_300k.update(
    max_steps=300000,

    # Using batch_size of 16 instead of 128 since we're training on 8 GPUs.
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
)

# samples/second = 3.778 (on one p3.2xlarge)
small_bert_50k = deepcopy(small_bert_100k)
small_bert_50k.update(
    max_steps=50000,
)

small_bert_100k_fp16 = dict(**small_bert_100k, fp16=True)
small_bert_50k_fp16 = dict(**small_bert_50k, fp16=True)


small_bert_large_dataset_100k = deepcopy(small_bert_100k)
small_bert_large_dataset_100k.update(
    max_steps=100000,
    max_seq_length=128,
    dataset_name=("wikipedia", "bookcorpus"),
    dataset_config_name=("20200501.en", None),
    tokenized_data_cache_dir="/mnt/datasets/huggingface/preprocessed-datasets/text",  # noqa: E501
    # added in 8/7 to fix discrepancies between dense and sparse
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16
)


# samples/second = 6.686 (on one p3.2xlarge)
mini_bert_100k = deepcopy(bert_100k)
mini_bert_100k.update(
    # Model Arguments
    model_type="bert",
    config_kwargs=dict(
        num_attention_heads=4,
        num_hidden_layers=4,
        hidden_size=256,  # hidden_size = 64 * num_attention_heads
        intermediate_size=1024,  # intermediate_size = 4 * hidden_size
        max_position_embeddings=128,
    ),
    # Training
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    # Dataset
    max_seq_length=128,
    dataset_name="wikipedia_plus_bookcorpus",
    dataset_config_name=None,
)

# samples/second = 6.676 (on one p3.2xlarge)
mini_bert_50k = deepcopy(mini_bert_100k)
mini_bert_50k.update(
    max_steps=50000,
)


# samples/second = 12.066 (on one p3.2xlarge)
tiny_bert_100k = deepcopy(bert_100k)
tiny_bert_100k.update(
    # Model Arguments
    model_type="bert",
    config_kwargs=dict(
        num_attention_heads=2,
        num_hidden_layers=2,
        hidden_size=128,  # hidden_size = 64 * num_attention_heads
        intermediate_size=512,  # intermediate_size = 4 * hidden_size
        max_position_embeddings=128,
    ),
    # Training
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    # Dataset
    max_seq_length=128,
    dataset_name="wikipedia_plus_bookcorpus",
    dataset_config_name=None,
)

# samples/second = 12.213 (on one p3.2xlarge)
tiny_bert_50k = deepcopy(tiny_bert_100k)
tiny_bert_50k.update(
    max_steps=50000,
)


CONFIGS = dict(
    nano_bert_debug=nano_bert_debug,

    small_bert_debug=small_bert_debug,
    mini_bert_debug=mini_bert_debug,
    tiny_bert_debug=tiny_bert_debug,

    small_bert_300k=small_bert_300k,

    small_bert_100k=small_bert_100k,
    mini_bert_100k=mini_bert_100k,
    tiny_bert_100k=tiny_bert_100k,

    small_bert_50k=small_bert_50k,
    mini_bert_50k=mini_bert_50k,
    tiny_bert_50k=tiny_bert_50k,

    small_bert_100k_fp16=small_bert_100k_fp16,
    small_bert_50k_fp16=small_bert_50k_fp16,
    small_bert_large_dataset_100k=small_bert_large_dataset_100k,
)
