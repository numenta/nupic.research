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
from functools import partial
from pprint import pformat

# FIXME: The experiments import Ray, but it must be imported before Pickle # noqa I001
import ray  # noqa: F401, I001
import torch.distributed
import transformers
from ray.tune.error import TuneError as TuneError
from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    HfArgumentParser,
    default_data_collator,
    set_seed,
)
from transformers.integrations import is_wandb_available
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from callbacks import TrackEvalMetrics
from experiments import CONFIGS
from integrations import (  # noqa I001
    CustomWandbCallback,
    init_ray_wandb_logger_callback,
)
from run_args import CustomTrainingArguments, DataTrainingArguments, ModelArguments
from run_utils import (
    TaskResults,
    check_best_metric,
    check_eval_and_max_steps,
    check_for_callback,
    check_hp_compute_objective,
    check_if_current_hp_best,
    check_mnli,
    check_rm_checkpoints,
    check_sparsity_callback,
    compute_objective,
    evaluate_language_model,
    evaluate_task_handler,
    get_best_run_and_link_best_predictions,
    get_labels,
    init_config,
    init_datasets_mlm,
    init_datasets_task,
    init_model,
    init_tokenizer,
    init_trainer,
    preprocess_datasets_mlm,
    preprocess_datasets_task,
    rm_prefixed_subdirs,
    run_hyperparameter_search,
    test_tasks,
    train,
    update_run_number,
)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def bold(text):
    """Bold inputed text for printing."""
    return "\033[1m" + text + "\033[0m"


def pdict(dictionary):
    """Pretty print dictionary."""
    return pformat(dictionary, indent=4)


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

        # Initialize wandb now to include the logs that follow.
        # For now, only support early wandb logging when running one experiment.
        distributed_initialized = torch.distributed.is_initialized()
        if is_wandb_available() and len(cmd_args.experiments) == 1:
            rank = -1 if not distributed_initialized else torch.distributed.get_rank()
            CustomWandbCallback.early_init(training_args, rank)

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
        logging.info(bold("\n\nRunning with experiment config:\n") + pdict(config_dict))

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

        logging.info(bold("\n\nTraining parameters:\n") + pdict(training_args.__dict__))
        logging.info(bold("\n\nModel parameters:\n") + pdict(model_args.__dict__))
        logging.info(bold("\n\nData parameters:\n") + pdict(data_args.__dict__))

        # Set seed before initializing model.
        set_seed(training_args.seed)
        logging.info(f"Seed to reproduce: {training_args.seed}")

        # Issue warnings if rm_checkpoints is not in the usual configuration
        check_rm_checkpoints(training_args, model_args)

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

    has_track_eval, metric_callback = check_for_callback(model_args, TrackEvalMetrics)

    # Run hp search or regular training
    if model_args.hp_num_trials >= 1:
        run_hyperparameter_search(
            model_args=model_args,
            config=config,
            tokenizer=tokenizer,
            data_collator=data_collator,
            training_args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
    else:
        trainer = init_trainer(
            tokenizer=tokenizer,
            data_collator=data_collator,
            training_args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            model=init_model(model_args, config, tokenizer),
            trainer_class=model_args.trainer_class,
            trainer_callbacks=model_args.trainer_callbacks or None,
        )
        if training_args.do_train:
            train(trainer,
                  training_args.output_dir,
                  training_args.rm_checkpoints,
                  last_checkpoint)

    # Evaluate in full eval dataset.
    # if using hp search, load best model before running evaluate
    if training_args.do_eval:
        logging.info("*** Evaluate ***")
        evaluate_language_model(trainer,
                                eval_dataset,
                                training_args.output_dir,
                                metric_callback)


def init_dataset_for_finetuning(model_args, data_args, training_args,
                                last_checkpoint=None):

    # TODO
    # edit multi_eval_sets so you can gather not just multiple eval sets
    # for a single task, but eval sets from multiple tasks
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
    check_sparsity_callback(model, model_args)
    check_mnli(model_args, data_args.task_name)
    # Tokenizing and preprocessing the datasets for downstream tasks
    # TODO: load from cached tokenized datasets for finetuning as well
    logging.info(f"Tokenizing datasets for finetuning ...")
    tokenized_datasets = preprocess_datasets_task(
        datasets, tokenizer, data_args,
        model, num_labels, label_list, is_regression
    )

    # Separate into train, eval and test
    train_dataset = tokenized_datasets["train"]

    # Allow multiple eval sets. For now, assume mnli is the only case
    eval_dataset = []
    if data_args.task_name == "mnli":
        if "eval_sets" in training_args.trainer_mixin_args:
            for eval_set in training_args.trainer_mixin_args["eval_sets"]:
                eval_dataset.append(tokenized_datasets[eval_set])
        else:
            eval_dataset.append(tokenized_datasets["validation_matched"])
    else:
        eval_dataset.append(tokenized_datasets["validation"])

    # If only one eval set, no need for a list
    if len(eval_dataset) == 1:
        eval_dataset = eval_dataset[0]

    test_dataset = None
    if (data_args.task_name is not None or data_args.test_file is not None):
        if training_args.do_predict:
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

    return (
        tokenizer, data_collator, train_dataset, eval_dataset, test_dataset, model,
        is_regression, tokenized_datasets, label_list, config
    )


def run_finetuning_single_task_with_hp_search(
    model_args, data_args, training_args, last_checkpoint=None
):
    """On a single task train, evaluate, and save results"""

    # Init dataset (same as without hp search)
    tokenizer, data_collator, train_dataset, eval_dataset, test_dataset, model, \
        is_regression, tokenized_datasets, label_list, config = \
        init_dataset_for_finetuning(
            model_args, data_args, training_args, last_checkpoint,
        )

    # Defines defaults required for hp search
    training_args.load_best_model_at_end = True
    training_args.disable_tqdm = True  # competes with ray output
    training_args.metric_for_best_model = model_args.hp_compute_objective[1]
    training_args.do_eval = False
    training_args.do_predict = False

    # Code safety run a second time due to training_args being changed above
    check_eval_and_max_steps(training_args, train_dataset)
    training_args = check_best_metric(training_args, data_args.task_name)
    model_args = check_hp_compute_objective(model_args,
                                            data_args.task_name,
                                            training_args)
    check_sparsity_callback(model, model_args)

    # Get fraction of the validation dataset to use in hp search
    if isinstance(eval_dataset, list):
        hp_eval_dataset = []
        for dataset in eval_dataset:
            if model_args.hp_validation_dataset_pct < 1:
                eval_set = dataset.shard(
                    index=1, num_shards=int(1 / model_args.hp_validation_dataset_pct)
                )
            else:
                eval_set = dataset
            hp_eval_dataset.append(eval_set)
    else:
        if model_args.hp_validation_dataset_pct < 1:
            hp_eval_dataset = eval_dataset.shard(
                index=1, num_shards=int(1 / model_args.hp_validation_dataset_pct)
            )
        else:
            hp_eval_dataset = eval_dataset

    # Specify how to re-init model each training run.
    def model_init():
        model_kwargs = dict(
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path, **model_kwargs
        )

        check_sparsity_callback(model, model_args)
        return model

    # Train
    trainer = init_trainer(
        tokenizer=tokenizer,
        data_collator=data_collator,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=hp_eval_dataset,
        trainer_class=model_args.trainer_class,  # changed
        trainer_callbacks=model_args.trainer_callbacks or None,
        model_init=model_init,  # changed
        task_name=data_args.task_name,  # does it matter?
        is_regression=is_regression,  # does it matter?
        finetuning=True  # see if it fixes key error issue
    )

    hp_search_kwargs = dict(
        direction=model_args.hp_compute_objective[0],
        backend="ray",
        n_trials=model_args.hp_num_trials,
        hp_space=model_args.hp_space,
        compute_objective=partial(
            compute_objective, objective=model_args.hp_compute_objective[1]
        ),
        local_dir=training_args.output_dir,
        resources_per_trial=model_args.hp_resources_per_trial,
        checkpoint_freq=0,
        keep_checkpoints_num=0,
        checkpoint_at_end=False,
    )

    # TODO
    # Get wandb to log properly
    # trainer.hyperparameter_search calls ray.tune()
    # you can set config or callbacks as a kwarg to trainer.hyperparameter_search
    # which gets passed to ray.tune

    # Update any extra kwargs defined in config
    hp_search_kwargs.update(**model_args.hp_extra_kwargs)

    # Run hp search and save results. Code to remove checkpoints won't get
    # called if ANY of the trials error out, so wrap with try/except.

    best_run = None
    try:
        best_run = trainer.hyperparameter_search(**hp_search_kwargs)
        logging.info(f"Best run: {best_run}")
    except TuneError:
        logging.info(f"One or more trials errored out")
    finally:
        # Make sure cleanup code gets called regardless of if hp search completes
        rm_prefixed_subdirs(training_args.output_dir, "run-")

    hp_res_file_name = f"best_run_results_{model_args.hp_compute_objective[1]}.txt"
    hp_res_file = os.path.join(training_args.output_dir, hp_res_file_name)
    # False if best_run stays as None
    write_new = check_if_current_hp_best(hp_res_file, model_args, best_run)

    if trainer.is_world_process_zero() and write_new:
        with open(hp_res_file, "w") as writer:
            writer.write("Hyperparameter search best run:\n")
            writer.write(f"run_id = {best_run.run_id}\n")
            writer.write(f"{training_args.metric_for_best_model}")
            writer.write(f"= {best_run.objective}\n")
            writer.write(f"\nHyperparameters:\n")
            for key, value in sorted(best_run.hyperparameters.items()):
                writer.write(f"{key} = {value}\n")

    # There is an existing issue on training multiple models in sequence in this code
    # There is a memory leakage on the model, a small amount of GPU memory remains after
    # the run and accumulates over several runs. It fails with OOM after about 20 runs,
    # even when all tensors on GPU are explicitly deleted, garbage is collected and
    # cache is cleared. Tried multiple solutions but this weird little hack is the only
    # thing that worked.
    model.to("cpu")

    return {}


def run_finetuning_single_task(
    model_args, data_args, training_args, last_checkpoint=None, run_idx=None,
):
    """On a single task train, evaluate, and save results"""

    tokenizer, data_collator, train_dataset, eval_dataset, test_dataset, model, \
        is_regression, tokenized_datasets, label_list, config = \
        init_dataset_for_finetuning(
            model_args, data_args, training_args, last_checkpoint
        )

    # Code safety
    check_eval_and_max_steps(training_args, train_dataset)
    training_args = check_best_metric(training_args, data_args.task_name)
    check_mnli(model_args, data_args.task_name)
    # Update where model is saved for each run
    training_args = update_run_number(training_args, run_idx)

    # Train
    trainer = init_trainer(
        tokenizer=tokenizer,
        data_collator=data_collator,
        training_args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        model=model,
        trainer_class=model_args.trainer_class,
        trainer_callbacks=model_args.trainer_callbacks or None,
        finetuning=True, task_name=data_args.task_name, is_regression=is_regression
    )

    if training_args.do_train:
        # Note, rm_checkpoints=True means one model will be saved
        # in the output_dir, and all checkpoint subdirectories will be
        # deleted when train() is called.
        train(trainer,
              training_args.output_dir,
              training_args.rm_checkpoints,
              last_checkpoint)

    eval_results = {}
    if training_args.do_eval:
        eval_results = evaluate_task_handler(
            trainer, data_args, model_args, training_args,
            eval_dataset, tokenized_datasets)

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

    has_track_eval, _ = check_for_callback(model_args, TrackEvalMetrics)
    if not has_track_eval:
        logging.warn(
            "You are running without tracking metrics throughout training."
            "This is strongly discouraged."
        )

    base_training_args = deepcopy(training_args)
    base_model_args = deepcopy(model_args)
    for task_name in data_args.task_names:
        data_args.task_name = task_name
        training_args = deepcopy(base_training_args)
        model_args = deepcopy(base_model_args)
        # For each task, save to a subfolder within run's root folder
        training_args.run_name = f"{base_training_args.run_name}_{task_name}"
        training_args.output_dir = os.path.join(
            base_training_args.output_dir, task_name
        )

        model_arg_keys = ["trainer_class", "task_hyperparams_proxy"]
        if task_name in model_args.task_hyperparams:
            for hp_key, hp_val in model_args.task_hyperparams[task_name].items():
                # maybe handle proxy task here
                if ("hp_" in hp_key) or hp_key in model_arg_keys:
                    setattr(model_args, hp_key, hp_val)
                else:
                    setattr(training_args, hp_key, hp_val)

        # These checks can change training args, which can affect TaskResults
        # attributes like metric_for_best_model
        training_args = check_best_metric(training_args, data_args.task_name)
        check_mnli(model_args, data_args.task_name)
        task_results = TaskResults(task_name, training_args)

        # Hack to ensure we don't do hp search num_runs times
        if model_args.hp_num_trials > 1:
            training_args.num_runs = 1

        # Run finetuning and save results
        for run_idx in range(training_args.num_runs):
            training_args.seed = random.randint(0, 1_000_000_000)
            set_seed(training_args.seed)

            if model_args.hp_num_trials > 1:
                eval_results = run_finetuning_single_task_with_hp_search(
                    model_args,
                    data_args,
                    training_args,
                    last_checkpoint=last_checkpoint
                )
            else:
                eval_results = run_finetuning_single_task(
                    model_args,
                    data_args,
                    training_args,
                    last_checkpoint=last_checkpoint,
                    run_idx=run_idx
                )

            task_results.append(eval_results)

        # Delete all finetuning run directories except for the best one
        # Ignore if this is a hyperparameter run, since the excess
        # is deleted within that function.
        if (model_args.hp_num_trials <= 1):
            best_run = get_best_run_and_link_best_predictions(
                training_args, task_results, task_name)
            skip = "run_" + best_run
            task_output_dir = os.path.dirname(training_args.output_dir)
            rm_prefixed_subdirs(task_output_dir, "run_", skip=skip)

        # If this is just a prediction run, ignore this block
        if training_args.do_eval:
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
