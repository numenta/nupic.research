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
# Table of results: (reported using a one p3.2xlarge)
#
# |------------------------------------------------------------------------------------|
# | model           | num params | loss  | perplexity | training time | samples/second |
# |-----------------|:----------:|:-----:|:----------:|:-------------:|:--------------:|
# | small_bert_100k | 27,523,072 | 2.951 | 14.532     | ~8 hrs        | 3.48           |
# | small_bert_50k  | 27,523,072 | 3.373 | 21.386     | ~2 hrs 40 min | 3.778          |
# | mini_bert_100k  | 10,615,808 | 3.669 | 27.342     | ~6 hrs        | 6.686          |
# | mini_bert_50k   | 10,615,808 | 4.464 | 56.309     | ~2 hrs        | 6.676          |
# | tiny_bert_100k  |  4,124,928 | 4.566 | 55.79276   | ~2 hrs 20 min | 12.066         |
# | tiny_bert_50k   |  4,124,928 | 5.990 | 330.234    | ~1 hrs        | 12.213         |
# |------------------------------------------------------------------------------------|


# samples/second = 2.536 p3.2xlarge
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


# samples/second = 3.989 p3.2xlarge
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


# samples/second = 5.428 p3.2xlarge
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

small_bert_50k = deepcopy(small_bert_100k)
small_bert_50k.update(
    max_steps=50000,
)


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

mini_bert_50k = deepcopy(mini_bert_100k)
mini_bert_50k.update(
    max_steps=50000,
)


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

tiny_bert_50k = deepcopy(tiny_bert_100k)
tiny_bert_50k.update(
    max_steps=50000,
)


CONFIGS = dict(
    small_bert_debug=small_bert_debug,
    mini_bert_debug=mini_bert_debug,
    tiny_bert_debug=tiny_bert_debug,

    small_bert_100k=small_bert_100k,
    mini_bert_100k=mini_bert_100k,
    tiny_bert_100k=tiny_bert_100k,

    small_bert_50k=small_bert_50k,
    mini_bert_50k=mini_bert_50k,
    tiny_bert_50k=tiny_bert_50k,
)
