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

import argparse
import logging
import os
import pickle
import random
import sys
from copy import deepcopy
from dataclasses import replace
from functools import partial
from pprint import pformat

import torch.distributed
import transformers
from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    DataCollatorWithPadding,
    HfArgumentParser,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from experiments import CONFIGS
from run_args import CustomTrainingArguments, DataTrainingArguments, ModelArguments
from run_utils import (
    TaskResults,
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
    test_tasks,
    train,
)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def main():
    cmd_parser = argparse.ArgumentParser()
    cmd_parser.add_argument("experiments", nargs="+", choices=list(CONFIGS.keys()),
                            help="Available experiments")
    cmd_parser.add_argument("--local_rank", default=None,
                            help="added by torch.distributed.launch")

    cmd_args = cmd_parser.parse_args()

    for experiment in cmd_args.experiments:
        config_dict = CONFIGS[experiment]
        local_rank = int(cmd_args.local_rank or -1)
        config_dict["local_rank"] = local_rank

        # See all possible arguments in transformers/training_args.py and ./run_args.py
        exp_parser = HfArgumentParser(
            (ModelArguments, DataTrainingArguments, CustomTrainingArguments)
        )
        model_args, data_args, training_args = exp_parser.parse_dict(config_dict)

        # Overrides default behavior of TrainingArguments of setting run name
        # equal to output_dir when not available
        if training_args.run_name == training_args.output_dir:
            training_args.run_name = experiment
        # Run name (or experiment name) is added to the output_dir
        training_args.output_dir = os.path.join(
            training_args.output_dir, training_args.run_name
        )

        if model_args.use_custom_wandbcallback:
            # Initialize wandb now to include the logs that follow.
            # For now, only support early wandb logging when running one experiment.
            from integrations import CustomWandbCallback
            if len(cmd_args.experiments) == 1:
                CustomWandbCallback.early_init(training_args, local_rank)

        # Detecting last checkpoint.
        last_checkpoint = None
        if (os.path.isdir(training_args.output_dir) and training_args.do_train
           and not training_args.overwrite_output_dir):
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
            logging.warning(f"Loading from checkpoint: {last_checkpoint} ")
            if (last_checkpoint is None
               and len(os.listdir(training_args.output_dir)) > 0):
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and "
                    "is not empty. Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None:
                logging.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To "
                    "avoid this behavior, change the `--output_dir` or add "
                    "`--overwrite_output_dir` to train from scratch."
                )

        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
            level=(logging.INFO if is_main_process(training_args.local_rank)
                   else logging.WARN)
        )

        # Log config.
        logging.info(f"Running with config:\n{pformat(config_dict, indent=4)}")

        # Log on each process the small summary:
        logging.warning(
            f"Process rank: {training_args.local_rank}, "
            f"device: {training_args.device}, n_gpu: {training_args.n_gpu} "
            f"distributed training: {bool(training_args.local_rank != -1)}, "
            f"16-bits training: {training_args.fp16}"
        )
        # Set the verbosity to info of the Transformers logging (on main process only):
        if is_main_process(training_args.local_rank):
            transformers.utils.logging.set_verbosity_info()
            transformers.utils.logging.enable_default_handler()
            transformers.utils.logging.enable_explicit_format()
        logging.info("Training/evaluation parameters %s", training_args)
        logging.info("Model parameters: %s", model_args)
        logging.info("Data parameters: %s", data_args)

        # Set seed before initializing model.
        set_seed(training_args.seed)
        logging.info(f"Seed to reproduce: {training_args.seed}")

        if model_args.finetuning:
            run_finetuning_multiple_tasks(
                model_args, data_args, training_args, last_checkpoint=last_checkpoint
            )
        else:
            run_pretraining(
                model_args, data_args, training_args, last_checkpoint=last_checkpoint
            )

        # destroy process group before launching another experiment
        if cmd_args.local_rank:
            torch.distributed.destroy_process_group()


def run_pretraining(
    model_args, data_args, training_args, last_checkpoint=None
):
    """Pretrain and evaluate a language model"""

    logging.info(f"Pre-training a masked language model.")

    datasets, tokenized_datasets, dataset_path = init_datasets_mlm(data_args)

    config = init_config(model_args)
    tokenizer = init_tokenizer(model_args)

    if tokenized_datasets is None:
        # Tokenizing and preprocessing the datasets for language modeling
        if training_args.do_train:
            column_names = datasets["train"].column_names
        else:
            column_names = datasets["validation"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        logging.info(f"Tokenizing datasets for pretraining ...")
        tokenized_datasets = preprocess_datasets_mlm(
            datasets, tokenizer, data_args,
            column_names, text_column_name
        )

        # Save only if a dataset_path has been defined in the previous steps
        # that will be True only when loading from dataset hub
        if data_args.save_tokenized_data and dataset_path is not None:
            logging.info(f"Saving tokenized dataset to {dataset_path}")
            tokenized_datasets.save_to_disk(dataset_path)

    # Separate into train, eval and test
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    # Log fingerprint used in HF smart caching
    logging.info(f"Dataset fingerprint: {train_dataset._fingerprint}")

    # Data collator will take care of randomly masking the tokens.
    # argument defined in experiment config
    assert hasattr(transformers, data_args.data_collator), \
        f"Data collator {data_args.data_collator} not available"
    data_collator = getattr(transformers, data_args.data_collator)(
        tokenizer=tokenizer, mlm_probability=data_args.mlm_probability
    )

    # Initialize distillation models
    if model_args.teacher_models_name_or_path:
        teacher_models = []
        for model_name_or_path in model_args.teacher_models_name_or_path:
            teacher_args = replace(model_args, model_name_or_path=model_name_or_path)
            teacher_config = init_config(teacher_args)
            teacher_models.append(init_model(teacher_args, teacher_config, tokenizer))

        model_args.trainer_extra_kwargs["teacher_models"] = teacher_models
        logging.info(f"{len(teacher_models)} teacher models initialized.")

    # Regular training run: init model, trainer, and call train
    if model_args.hp_num_trials <= 1:

        trainer = init_trainer(
            tokenizer=tokenizer,
            data_collator=data_collator,
            training_args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            model=init_model(model_args, config, tokenizer),
            trainer_class=model_args.trainer_class,
            trainer_extra_kwargs=model_args.trainer_extra_kwargs,
            trainer_callbacks=model_args.trainer_callbacks or None,
        )

        if training_args.do_train:
            train(trainer, training_args.output_dir, last_checkpoint)

    # Alternative: using hp search
    else:
        # Extra trainer configurations required for hpsearch
        model_args.trainer_extra_kwargs.update(
            model_init=partial(init_model, model_args, config, tokenizer, False)
        )
        training_args.load_best_model_at_end = True
        training_args.disable_tqdm = True
        # TODO: sync metric_for_best_model with compute_objective, should be same metric
        # and accept custom metrics defined by user
        training_args.metric_for_best_model = "eval_loss"
        # TODO: load best model and proceed to evaluation normally after hp search
        training_args.do_eval = False
        training_args.do_predict = False

        # Get fraction of the validation dataset to use in hp search
        hp_eval_dataset = eval_dataset.shard(
            index=1, num_shards=int(1 / model_args.hp_validation_dataset_pct)
        )

        trainer = init_trainer(
            tokenizer=tokenizer,
            data_collator=data_collator,
            training_args=training_args,
            train_dataset=train_dataset,
            eval_dataset=hp_eval_dataset,
            trainer_class=model_args.trainer_class,
            trainer_extra_kwargs=model_args.trainer_extra_kwargs,
            trainer_callbacks=model_args.trainer_callbacks or None,
        )

        hp_search_kwargs = dict(
            direction="maximize",
            backend="ray",
            n_trials=model_args.hp_num_trials,
            hp_space=model_args.hp_space,
            compute_objective=model_args.hp_compute_objective,
            local_dir=training_args.output_dir,
            resources_per_trial=dict(
                cpu=os.cpu_count() / torch.cuda.device_count() - 1,
                gpu=1
            ),
            checkpoint_freq=0,
            keep_checkpoints_num=0,
            checkpoint_at_end=False,
        )
        # Update any extra kwargs defined in config
        hp_search_kwargs.update(**model_args.hp_extra_kwargs)

        # Run hp search and save results
        best_run = trainer.hyperparameter_search(**hp_search_kwargs)
        logging.info(f"Best run: {best_run}")

        hp_res_file = os.path.join(training_args.output_dir, "hp_search_results.txt")
        if trainer.is_world_process_zero():
            with open(hp_res_file, "w") as writer:
                writer.write("Hyperparameter search best run:\n")
                writer.write(f"run_id = {best_run.run_id}\n")
                writer.write(f"{training_args.metric_for_best_model}")
                writer.write(f"= {best_run.objective}\n")
                writer.write(f"\nHyperparameters:\n")
                for key, value in sorted(best_run.hyperparameters.items()):
                    writer.write(f"{key} = {value}\n")

    # Evaluate in full eval dataset.
    # if using hp search, load best model before running evaluate
    if training_args.do_eval:
        logging.info("*** Evaluate ***")
        evaluate_language_model(trainer, eval_dataset, training_args.output_dir)


def run_finetuning_single_task(
    model_args, data_args, training_args, last_checkpoint=None
):
    """On a single task train, evaluate, and save results"""

    datasets = init_datasets_task(data_args, training_args)
    is_regression, label_list, num_labels = get_labels(datasets, data_args)
    logging.info(f"Training {data_args.task_name} with {num_labels} labels")

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
    logging.info(f"Tokenizing datasets for finetuning ...")
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
    logging.info(f"Dataset fingerprint: {train_dataset._fingerprint}")

    # Data collator will default to DataCollatorWithPadding,
    # so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Train
    trainer = init_trainer(
        tokenizer=tokenizer,
        data_collator=data_collator,
        training_args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        model=init_model(model_args, config, tokenizer),
        trainer_callbacks=model_args.trainer_callbacks or None,
        finetuning=True, task_name=data_args.task_name, is_regression=is_regression
    )
    if training_args.do_train:
        train(trainer, training_args.output_dir, last_checkpoint)

    # Evaluate
    eval_results = {}
    if training_args.do_eval:
        logging.info("*** Evaluate ***")

        # Handle special case of extra validation dataset for MNLI
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(tokenized_datasets["validation_mismatched"])

        eval_results = evaluate_tasks(
            trainer, training_args.output_dir, tasks, eval_datasets
        )

    # Test/Predict
    if training_args.do_predict:
        logging.info("*** Test ***")

        # Handle special case of extra test dataset for MNLI
        tasks = [data_args.task_name]
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            test_datasets.append(tokenized_datasets["test_mismatched"])

        test_tasks(
            trainer, training_args.output_dir, tasks, test_datasets,
            is_regression, label_list
        )

    # There is an existing issue on training multiple models in sequence in this code
    # There is a memory leakage on the model, a small amount of GPU memory remains after
    # the run and accumulates over several runs. It fails with OOM after about 20 runs,
    # even when all tensors on GPU are explicitly deleted, garbage is collected and
    # cache is cleared. Tried multiple solutions but this weird little hack is the only
    # thing that worked.
    model.to("cpu")

    return eval_results


def run_finetuning_multiple_tasks(
    model_args, data_args, training_args, last_checkpoint=None
):
    """Loop through all tasks, train, evaluate, and save results"""

    logging.info(f"Finetuning model for downstream tasks.")

    # If results file already exists, open it and only update keys
    results_path = os.path.join(training_args.output_dir, "task_results.p")
    if os.path.exists(results_path) and not data_args.override_finetuning_results:
        logging.info(f"Updating existing task_results in {results_path}")
        with open(results_path, "rb") as f:
            results = pickle.load(f)
    else:
        results = {}

    base_training_args = deepcopy(training_args)
    for task_name in data_args.task_names:
        data_args.task_name = task_name
        training_args = deepcopy(base_training_args)
        # For each task, save to a subfolder within run's root folder
        training_args.run_name = f"{base_training_args.run_name}_{task_name}"
        training_args.output_dir = os.path.join(
            base_training_args.output_dir, task_name
        )
        # Update any custom training hyperparameter
        # TODO: allow hyperparameter search for each task
        if task_name in model_args.task_hyperparams:
            for hp_key, hp_val in model_args.task_hyperparams[task_name].items():
                setattr(training_args, hp_key, hp_val)
        # create Task Results
        task_results = TaskResults(task_name, training_args=training_args)

        # Run finetuning and save results
        for _ in range(training_args.num_runs):
            # reset seed per run
            training_args.seed = random.randint(0, 1000000000)
            set_seed(training_args.seed)

            eval_results = run_finetuning_single_task(
                model_args, data_args, training_args, last_checkpoint=last_checkpoint
            )
            task_results.append(eval_results)

        task_results.reduce_metrics(reduction="mean")
        logging.info(f"{task_name} results: {task_results.to_string()}")
        logging.info(f"{task_name} consolidated: {task_results.consolidate()}")
        results[task_name] = task_results

        # Pickle and save results
        if is_main_process(base_training_args.local_rank):
            logging.info(f"Saving task_results to {results_path}")
            with open(results_path, "wb") as file:
                pickle.dump(results, file)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
