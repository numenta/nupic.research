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
from copy import deepcopy
import random

transformers_base = dict(

    # Model arguments
    cache_dir="/mnt/efs/results/pretrained-models/huggingface",
    use_fast_tokenizer=True,
    use_auth_token=False,

    # Data training arguments
    output_dir=os.path.expanduser("~/nta/results/experiments/transformers/transformers_base"),
    overwrite_output_dir=True,
    overwrite_cache=False,

    # Training Arguments
    run_name="transformers_base",
    seed=random.randint(0, 1000000),
    do_train=True,
    do_eval=True,
    num_train_epochs=3  # is overriden if max_steps is defined

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
    run_name="bert_base",
    output_dir=os.path.expanduser("~/nta/results/experiments/transformers/bert_base"),
    overwrite_output_dir=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=1e-4,
    lr_scheduler_type="linear",
    warmup_steps=500,
    weight_decay=1e-6,
    max_steps=1000000,

)

debug_bert = deepcopy(bert_base)
debug_bert.update(

    # Training Arguments
    run_name="debug_run",
    logging_first_step=True,
    logging_steps=30,  # also define eval_steps, redundant
    eval_steps=30,
    warmup_steps=30,
    max_steps=600,
    disable_tqdm=False,  # default False
    output_dir=os.path.expanduser("~/nta/results/experiments/transformers/debug_bert"),
    overwrite_output_dir=True,

)

# Export configurations in this file
CONFIGS = dict(
    transformers_base=transformers_base,
    bert_base=bert_base,
    debug_bert=debug_bert,
)
