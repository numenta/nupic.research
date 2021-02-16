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
Pre-train or finetune HuggingFace models for masked language modeling
(BERT, ALBERT, RoBERTa...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=masked-lm

Adapted from:
github.com/huggingface/transformers/blob/master/examples/language-modeling/run_mlm.py
github.com/huggingface/transformers/blob/master/examples/text-classification/run_glue.py
"""

import logging
import os
import sys

import transformers
from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    DataCollatorWithPadding,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from experiments import CONFIGS
from run_args import DataTrainingArguments, ModelArguments
from run_utils import (
    evaluate_language_model,
    evaluate_tasks,
    get_labels,
    init_config,
    init_datasets_mlm,
    init_datasets_task,
    init_model,
    init_tokenizer,
    init_trainer,
    preprocess_datasets_mlm,
    preprocess_datasets_task,
    train,
)

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )

    # Add option to run from local experiment file
    parser.add_argument("-e", "--experiment", dest="experiment",
                        choices=list(CONFIGS.keys()),
                        help="Available experiments")
    args = parser.parse_args()

    trainer_callbacks = None
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    elif "experiment" in args:
        # If experiment given, use arguments from experiment file instead
        config_dict = CONFIGS[args.experiment]
        config_dict["local_rank"] = args.local_rank  # added by torch.distributed.launch
        model_args, data_args, training_args = parser.parse_dict(config_dict)
        # Callbacks can only be passed when using experiment
        if "trainer_callbacks" in config_dict:
            trainer_callbacks = config_dict["trainer_callbacks"]
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if (os.path.isdir(training_args.output_dir) and training_args.do_train
       and not training_args.overwrite_output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        logger.warning(f"Loading from checkpoint: {last_checkpoint} ")
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and "
                "is not empty. Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To "
                "avoid this behavior, change the `--output_dir` or add "
                "`--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank)
                    else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, "
        f"device: {training_args.device}, n_gpu: {training_args.n_gpu} "
        f"distributed training: {bool(training_args.local_rank != -1)}, "
        f"16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters: %s", model_args)
    logger.info("Data parameters: %s", data_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)
    logger.info(f"Seed to reproduce: {training_args.seed}")

    if model_args.finetuning:
        logger.info(f"Finetuning model for downstream tasks.")
        run_finetuning(
            model_args, data_args, training_args,
            trainer_callbacks=trainer_callbacks, last_checkpoint=last_checkpoint
        )
    else:
        logger.info(f"Pre-training a masked language model.")
        run_pretraining(
            model_args, data_args, training_args,
            trainer_callbacks=trainer_callbacks, last_checkpoint=last_checkpoint
        )


def run_pretraining(model_args, data_args, training_args,
                    trainer_callbacks=None, last_checkpoint=None):

    datasets, tokenized_datasets, dataset_path = init_datasets_mlm(data_args)

    config = init_config(model_args, extra_config_kwargs=None)
    tokenizer = init_tokenizer(model_args)
    model = init_model(model_args, config, tokenizer, finetuning=False)

    if tokenized_datasets is None:
        # Tokenizing and preprocessing the datasets for language modeling
        if training_args.do_train:
            column_names = datasets["train"].column_names
        else:
            column_names = datasets["validation"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        logger.info(f"Tokenizing datasets for pretraining ...")
        tokenized_datasets = preprocess_datasets_mlm(
            datasets, tokenizer, data_args,
            column_names, text_column_name
        )

        # Save only if a dataset_path has been defined in the previous steps
        # that will be True only when loading from dataset hub
        if data_args.save_tokenized_data and dataset_path is not None:
            logger.info(f"Saving tokenized dataset to {dataset_path}")
            tokenized_datasets.save_to_disk(dataset_path)

    # Separate into train, eval and test
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    # Log fingerprint used in HF smart caching
    logger.info(f"Dataset fingerprint: {train_dataset._fingerprint}")

    # Data collator will take care of randomly masking the tokens.
    # argument defined in experiment config
    assert hasattr(transformers, data_args.data_collator), \
        f"Data collator {data_args.data_collator} not available"
    data_collator = getattr(transformers, data_args.data_collator)(
        tokenizer=tokenizer, mlm_probability=data_args.mlm_probability
    )

    # Train and evaluate
    trainer = init_trainer(
        model, tokenizer, data_collator, training_args,
        train_dataset, eval_dataset, trainer_callbacks,
    )
    train(trainer, training_args, model_args, last_checkpoint)
    evaluate_language_model(trainer, training_args)


def run_finetuning(model_args, data_args, training_args,
                   trainer_callbacks=None, last_checkpoint=None):

    datasets = init_datasets_task(data_args, training_args)
    is_regression, label_list, num_labels = get_labels(datasets, data_args)

    # For finetuning required to add labels and task name to config kwargs
    extra_config_kwargs = dict(
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
    )
    config = init_config(model_args, extra_config_kwargs=extra_config_kwargs)
    tokenizer = init_tokenizer(model_args)
    model = init_model(model_args, config, tokenizer, finetuning=True)

    # Tokenizing and preprocessing the datasets for downstream tasks
    # TODO: load from cached tokenized datasets for finetuning as well
    logger.info(f"Tokenizing datasets for finetuning ...")
    tokenized_datasets = preprocess_datasets_task(
        datasets, tokenizer, data_args,
        model, num_labels, label_list, is_regression
    )

    # Separate into train, eval and test
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets[
        "validation_matched" if data_args.task_name == "mnli" else "validation"
    ]
    test_dataset = None
    if ((data_args.task_name is not None or data_args.test_file is not None)
       and training_args.do_predict):
        test_dataset = tokenized_datasets[
            "test_matched" if data_args.task_name == "mnli" else "test"
        ]

    # Log fingerprint used in HF smart caching
    logger.info(f"Dataset fingerprint: {train_dataset._fingerprint}")

    # Data collator will default to DataCollatorWithPadding,
    # so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Train and evaluate
    trainer = init_trainer(
        model, tokenizer, data_collator, training_args,
        train_dataset, eval_dataset, trainer_callbacks,
        finetuning=True, task_name=data_args.task_name, is_regression=is_regression
    )
    train(trainer, training_args, model_args, last_checkpoint)
    evaluate_tasks(
        trainer, training_args, data_args, datasets,
        eval_dataset, test_dataset, is_regression, label_list
    )


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
