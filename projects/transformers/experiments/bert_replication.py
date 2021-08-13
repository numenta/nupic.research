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
from copy import deepcopy

from transformers import Trainer

from trainer_mixins import LRRangeTestMixin

from .base import transformers_base


"""
Version of BERT that mimics hyperparams in the original paper
Any intentional differences are highlighted below

dataset:
- Original: 2.5 mi words on bookcorpus and 5mi words on wikipedia.
- This: bookcorpus is similar, however wikipedia is at least 5x larger based
on the dataset size. Text pre-processing is the same as in original BERT repo, but
the data wrangling script to select the specific 5mi words is not available.

seq_length:
- Original: seq_length 128 for 900k steps, seq_length 512 for extra steps
- This: seq_length for 100k steps

max_steps:
- Original: 1mi; This: 100k

warmup_steps:
- Original: 10k; This: 2k

data_collator:
- Original: uses DataCollatorForLanguageModeling. Can mask parts of words.
- This: masks only whole words, not individual tokens. Updated Bert repo
shows better results on GLUE with this approach.

tokenizer_name (cased vs uncased):
- Original: unclear if original BERT uses cased or uncased. Cased used when
case information is important for task - e.g., Named Entity Recognition (NER)
or Part-of-Speech tagging (POS).
- This: cased. Updated BERT repo shows better results on GLUE using cased

adam_epsilon:
- Original: unclear in paper, but likely 1e-8 from other sources. RoBERTa uses 1e-6
- This: 1e-8. Consider using RoBERTa setting when using larger batch size

Other relevant notes:
* Dropout=0.1 and activation=gelu are part of default BertConfig, loaded when
model_type is set to 'bert'
* LR Scheduler: after warmup, it decays to zero. lr_scheduler_type change
how it decays (e.g cosine, cosine_with_restarts, polynomial, or constant/no decay).
* Batch size is determined by the number of GPUS used. Original bs is 256. Given
per_device_train_batch_size=8 it should be trained on 32 GPUs to replicate bs
* Some data training arguments are unclear in original BERT. Using
default values: max_seq_length=None, line_by_line=False, pad_to_max_length=False,
"""


class LRRangeTestTrainer(LRRangeTestMixin,
                         Trainer):
    pass


lr_range_test_args = dict(
    max_steps=100,
    trainer_class=LRRangeTestTrainer,
    trainer_mixin_args=dict(
        # LR Range Test
        min_lr=0.0001,
        max_lr=0.005,
        test_mode="linear",

        # KD
        teacher_model_names_or_paths=[
            "/mnt/efs/results/pretrained-models/transformers-local/bert_1mi"
        ],
    ),
    overwrite_output_dir=True,
)

bert_100k = deepcopy(transformers_base)
bert_100k.update(

    # Model arguments
    model_type="bert",
    model_revision="main",
    tokenizer_name="bert-base-cased",

    #  Data Training arguments
    dataset_name=("wikipedia", "bookcorpus"),
    dataset_config_name=("20200501.en", None),
    mlm_probability=0.15,
    validation_split_percentage=5,
    max_seq_length=None,
    line_by_line=False,
    pad_to_max_length=False,
    data_collator="DataCollatorForWholeWordMask",

    # Training Arguments
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=1e-4,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    weight_decay=1e-2,
    warmup_steps=2000,
    max_steps=100000,
    lr_scheduler_type="linear",

    # Training Arguments - checkpointing
    logging_steps=500,
    logging_first_step=True,
    save_steps=5000,
    save_total_limit=5,
    overwrite_output_dir=False,

    # speeding up
    # fp16=True,
    dataloader_num_workers=4,

)


# This is an lr-range test for bert_100k
# Results here: https://wandb.ai/nupic-research/huggingface/runs/1gtfjmw5
# Suggested max_lr=0.00084
bert_lr_range_test = deepcopy(bert_100k)
bert_lr_range_test.update(
    tokenized_data_cache_dir="/mnt/datasets/huggingface/preprocessed-datasets/text",
    **lr_range_test_args
)


# This is an lr-range test for bert_100k trained with fp16
# Results here: https://wandb.ai/nupic-research/huggingface/runs/2f9sga1a
# Suggested max_lr=0.00079
bert_fp16_lr_range_test = deepcopy(bert_100k)
bert_fp16_lr_range_test.update(
    tokenized_data_cache_dir="/mnt/datasets/huggingface/preprocessed-datasets/text",
    **lr_range_test_args
)


# Equivalent to bert 100k but trained with 2K batch size for 12.5K steps using
# deepspeed on 4 x p3dn.24xlarge, for a total of 32 GPUs with 32Gb each GPU.
# It takes 7h with the final eval_loss of 2.225.
bert_100k_deepspeed_bsz_2k = deepcopy(bert_100k)
bert_100k_deepspeed_bsz_2k.update(
    # tokenized_data_cache_dir="/mnt/efs/results/preprocessed-datasets/text",
    tokenized_data_cache_dir="/mnt/datasets/huggingface/preprocessed-datasets/text",

    # Training Arguments
    gradient_accumulation_steps=2,
    per_device_train_batch_size=32,  # Requires 26Gb GPU memory
    per_device_eval_batch_size=32,
    warmup_steps=500,
    learning_rate=2e-4,
    max_steps=12500,

    # Training Arguments - checkpointing
    logging_steps=100,
    save_steps=500,

    deepspeed={
        "zero_optimization": {
            "stage": 1,
        },
        "fp16": {
            "enabled": True,
            "initial_scale_power": 14,
        },
        "gradient_clipping": 1.0,
        "sparse_gradients": True,
        "steps_per_print": 100,
    },
)

bert_1mi = deepcopy(bert_100k)
bert_1mi.update(
    max_steps=1000000,
)

# Export configurations in this file
CONFIGS = dict(
    bert_lr_range_test=bert_lr_range_test,
    bert_fp16_lr_range_test=bert_fp16_lr_range_test,
    bert_100k_deepspeed_bsz_2k=bert_100k_deepspeed_bsz_2k,
    bert_100k=bert_100k,
    bert_1mi=bert_1mi
)
