#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see htt"://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#

"""
Base Transformers Experiment configuration.
"""

import os
import random
from copy import deepcopy

transformers_base = dict(

    # Model arguments
    cache_dir="/mnt/efs/results/pretrained-models/huggingface",
    # tokenized_data_cache_dir="/mnt/efs/results/preprocessed-datasets/text",
    tokenized_data_cache_dir="/mnt/datasets/huggingface/preprocessed-datasets/text",  # noqa: E501
    reuse_tokenized_data=True,
    save_tokenized_data=True,
    use_fast_tokenizer=True,
    use_auth_token=False,

    # Data training arguments
    output_dir=os.path.expanduser("~/nta/results/experiments/transformers"),  # noqa: E501
    overwrite_output_dir=True,
    overwrite_cache=False,

    # Training Arguments
    seed=random.randint(0, 1000000000),
    do_train=True,
    do_eval=True,
    dataloader_drop_last=True,  # keeps consistent batch size
    num_train_epochs=3,  # is overridden if max_steps is defined
    ddp_find_unused_parameters=False,  # Should not have any unused parameters
)

bert_base = deepcopy(transformers_base)
bert_base.update(

    # Model arguments
    model_type="bert",
    model_revision="main",
    tokenizer_name="bert-base-cased",

    #  Data Training arguments
    dataset_name="wikitext",
    dataset_config_name="wikitext-2-raw-v1",
    mlm_probability=0.15,
    validation_split_percentage=5,
    max_seq_length=None,
    line_by_line=False,
    pad_to_max_length=False,

    # Training Arguments
    overwrite_output_dir=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=1e-4,
    lr_scheduler_type="linear",
    warmup_steps=500,
    weight_decay=1e-6,
    max_steps=1000000,

)

"""Configs used to debug single and multiple datasets loading and training"""
debug_bert = deepcopy(bert_base)
debug_bert.update(

    #  Data Training arguments
    dataset_name="wikitext",
    dataset_config_name="wikitext-2-raw-v1",

    # Training Arguments
    logging_first_step=True,
    logging_steps=30,  # also define eval_steps, redundant
    eval_steps=30,
    warmup_steps=30,
    max_steps=90,
    disable_tqdm=False,  # default False
    overwrite_output_dir=True,

)

debug_bert_multiple_data = deepcopy(debug_bert)
debug_bert_multiple_data.update(

    # Training Arguments
    dataset_name=("wikitext", "ptb_text_only"),
    dataset_config_name=("wikitext-2-raw-v1", None),

)

"""Only load and tokenize english version of wikipedia and Toronto BookCorpus"""
full_data_load = deepcopy(debug_bert)
full_data_load.update(

    # Training Arguments
    dataset_name=("wikipedia", "bookcorpus"),
    dataset_config_name=("20200501.en", None),
    do_train=False,
    do_eval=False,

)


# Export configurations in this file
CONFIGS = dict(
    transformers_base=transformers_base,
    bert_base=bert_base,
    debug_bert=debug_bert,
    debug_bert_multiple_data=debug_bert_multiple_data,
    full_data_load=full_data_load,
)
