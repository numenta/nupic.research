#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see http://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#

"""
Adapted from huggingface/transformers/blob/master/examples/language-modeling/run_mlm.py
plus other sources

Libraries required are `transformers` and `datasets`, both can be installed with pip

To reuse cached datasets, set the following environment variable prior to running:
HF_HOME=/mnt/efs/results/cache/huggingface
"""

import argparse
import os
import random

from datasets import concatenate_datasets, load_dataset
from datasets.dataset_dict import DatasetDict
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    BertConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

from training_utils import run_hf, run_ray_distributed, run_ray_single_instance


def main(train_function):

    # ----- Parse local_rank for torch.distributed.launch -----------

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    local_rank = parser.parse_args().local_rank
    if local_rank is None:
        local_rank = 0

    # ----- Configurable Params -----------

    # List of dicts with configuration for each dataset to be loaded
    # see available datasets in the Hub: https://huggingface.co/datasets. sizes
    # are of generated dataset, can be an order of magnitude larger after tokenization.
    # Not all datasets can be concatenated without preprocessing, features must align
    datasets_args = [
        dict(path="wikitext", name="wikitext-2-raw-v1"),  # 12.91 MB
        # dict(path="wikitext", name="wikitext-103-raw-v1"),  # 524 MB
        # dict(path="ptb_text_only"), # 5.7 MB
        # dict(path="bookcorpus"),  # 4.63 GB
        # dict(path="wikipedia"),  # 35.38 GB
    ]

    # Training params
    # note: in V100 bs=8 uses 11/16 of available gpu mem, bs=12 uses 15/16

    output_dir = os.path.expanduser("~/nta/results/bert")
    training_args = TrainingArguments(
        # Logging
        output_dir=output_dir,
        logging_first_step=True,
        logging_steps=10,  # also define eval_steps
        eval_steps=10,  # is it evaluating? where are results?
        max_steps=30,
        disable_tqdm=True,
        run_name="debug_run",  # used for wandb, not for Ray
        # hyperparams
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=1e-4,
        lr_scheduler_type="linear",
        warmup_steps=500,
        weight_decay=1e-6,
        # train epochs replaced by steps
        # num_train_epochs=3,
        # logging_dir=None
    )

    # Evaluate refers to evaluating perplexity on trained model in the validation set
    # doesn't refer to finetuning and evaluating on downstream tasks such as GLUE
    seed = random.randint(0, 1000000)

    # Changing the tokenizer will result in re-tokenizing the dataset.
    # As a reference, BERT tokenization will take ~ 3 hours for a 5GB dataset
    config_class = BertConfig
    tokenizer_name = "bert-base-cased"

    # ----- Seed -----------

    set_seed(seed)
    print(f"Seed to reproduce: {seed}")

    # ----- Dataset -----------

    # Load multiple datasets and concatenate.
    # using only 'train' and 'validation' sets, could also include 'test'
    # if no split is defined, load_dataset returns DatasetDict with all available splits
    train_datasets = [load_dataset(**args, split="train") for args in datasets_args]
    val_datasets = [load_dataset(**args, split="validation") for args in datasets_args]

    dataset = DatasetDict()
    dataset["train"] = concatenate_datasets(train_datasets)
    dataset["validation"] = concatenate_datasets(val_datasets)

    def load_and_split_dataset(dataset_args, split_percentage=5):
        """Alternative: if no validation set available, manuallly split the train set"""

        dataset = DatasetDict()
        dataset["train"] = load_dataset(
            **dataset_args, split=f"train[{split_percentage}%:]"
        )
        dataset["validation"] = load_dataset(
            **dataset_args, split=f"train[:{split_percentage}%]"
        )
        return dataset

    # ----- Load Model -----------

    # Load model
    config = config_class()
    model = AutoModelForMaskedLM.from_config(config)

    # Load tokenizer
    # use_fast falls back to tokenizer lib implementation under the hood
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    model.resize_token_embeddings(len(tokenizer))

    # ----- Preprocess dataset -----------

    # Only use the text column name when doing language modeling
    # this feature might have a different name depending on the dataset
    # might need to change column names prior to concatenating, if that is the case
    column_names = dataset["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    # Setting overwrite_cache to False will retokenize the dataset.
    # do not overwrite cache if using shared cache repository.
    overwrite_cache = False
    preprocessing_num_workers = None

    # We tokenize every text, then concatenate them together before splitting in smaller
    # parts. We use `return_special_tokens_mask=True` given
    # DataCollatorForLanguageModeling is more efficient when it
    # receives the `special_tokens_mask`.
    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        num_proc=preprocessing_num_workers,
        load_from_cache_file=not overwrite_cache,
    )

    # Main data processing function that will concatenate all texts from our dataset and
    # generate chunks of max_seq_length.
    max_seq_length = tokenizer.model_max_length

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it
        # instead of this drop, you can customize this part to your needs.
        total_length = (total_length // max_seq_length) * max_seq_length
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + max_seq_length] for i in
                range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so
    # group_texts throws away a remainder for each of those groups of 1,000 texts.
    # You can adjust batch_size here but a higher value will be slower to preprocess.
    tokenized_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        num_proc=preprocessing_num_workers,
        load_from_cache_file=not overwrite_cache,
    )

    # Data collator
    # This one will take care of randomly masking the tokens.
    # Q: what about dynamic masking, used in Roberta?
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=0.15
    )

    # ----- Setup Trainer -----------

    # Initialize Trainer. Similar to Vernon's Experiment class.
    # dataloader and training loop are contained in Trainer abstraction
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # ----- Functions to train and evaluate -----------

    if train_function == "huggingface":
        # Tested
        run_hf(trainer, output_dir, local_rank, save_model=True, evaluate=True)

    elif train_function == "ray_single_node":
        # Tested
        run_ray_single_instance(
            trainer,
            name="bert_test",
            config=None,
            num_samples=1,
            local_dir=os.path.expanduser("~/nta/results/experiments/transformers"),
            keep_checkpoints_num=1,
            resources_per_trial={"cpu": 8},
            # note: checkpoint arguments cannot be used with a checkpointable function
        )

    elif train_function == "ray_multiple_nodes":
        # Untested
        run_ray_distributed(
            trainer,
            name="bert_test",
            config=None,
            num_samples=1,
            local_dir=os.path.expanduser("~/nta/results/experiments/transformers"),
            keep_checkpoints_num=1,
            queue_trials=True,
            verbose=2,
            resources_per_trial={"gpu": 4},
        )


if __name__ == "__main__":
    main(train_function="huggingface")
