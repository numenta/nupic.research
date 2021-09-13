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
Auxiliary functions to run.py file
"""

import logging
import math
import multiprocessing
import os
import pickle
import shutil
from collections import Counter, defaultdict
from functools import partial
from hashlib import blake2b

import numpy as np
import pandas as pd
from datasets import concatenate_datasets, load_dataset, load_from_disk
from datasets.dataset_dict import DatasetDict
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, matthews_corrcoef
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    PretrainedConfig,
    Trainer,
    TrainerCallback,
)

from callbacks import RezeroWeightsCallback, TrackEvalMetrics
from finetuning_constants import (
    GLUE_NAMES_PER_TASK,
    RAW_REPORTING_METRICS_PER_TASK,
    REPORTING_METRICS_PER_TASK,
)
from nupic.research.frameworks.pytorch.model_utils import count_nonzero_params
from nupic.torch.modules.sparse_weights import SparseWeightsBase

__all__ = [
    "evaluate_language_model",
    "evaluate_task",
    "get_labels",
    "init_config",
    "init_datasets_mlm",
    "init_datasets_task",
    "init_model",
    "init_tokenizer",
    "init_trainer",
    "preprocess_datasets_mlm",
    "preprocess_datasets_task",
    "run_hyperparameter_search",
    "test_tasks",
    "train",
]

"""
GLUE:
- SentencePair: (entailment, question answering)
    - 2 classes: QQP (question pairing), QNLI (adapted q&A), MRPC (paraphrase),
      RTE (adapted entailment from 3 to 2), WNLI (adapted multiple choice)
    - 3 classes: MNLI (entailment, neutral or contradiction)
    - 1 variable regresssion: STS-B  (similarity, from 0 to 5)
- Single Sentence, 2 classes: SST-2 (sentiment), COLA (grammatically correct)
"""

# Tasks ordered by training time
TASK_TO_KEYS = {
    "wnli": ("sentence1", "sentence2"),
    "rte": ("sentence1", "sentence2"),
    "stsb": ("sentence1", "sentence2"),
    "mrpc": ("sentence1", "sentence2"),
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "mnli": ("premise", "hypothesis"),  # includes matched and mismatched
}


def train(trainer, output_dir, rm_checkpoints, last_checkpoint=None):
    """Trainig function applicable to pretraining language models and finetuning."""

    logging.info("Before training: total params: {:,} non zero params: {:,}".format(
        *count_nonzero_params(trainer.model)
    ))

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload

    output_train_file = os.path.join(output_dir, "train_results.txt")
    if trainer.is_world_process_zero():
        with open(output_train_file, "w") as writer:
            logging.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logging.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

        # Need to save the state, since Trainer.save_model saves only the tokenizer
        # with the model
        trainer.state.save_to_json(
            os.path.join(output_dir, "trainer_state.json")
        )

    logging.info("After training: total params: {:,} non zero params: {:,}".format(
        *count_nonzero_params(trainer.model)
    ))

    if rm_checkpoints:
        rm_prefixed_subdirs(output_dir, "checkpoint-")


def evaluate_task(trainer, output_dir, task, eval_dataset):
    """
    Evaluate on one task after finetuning. If mnli or if MultiEvalSetTrainer,
    mixin will handle evaluating on multiple sets for you.
    Returns evaluation dict with results.
    """
    eval_results = {}

    # if mnli, special trainer instance handles loop for you
    if "MultiEvalSetTrainer" in str(type(trainer)):
        eval_result = trainer.evaluate()
    else:
        eval_result = trainer.evaluate(eval_dataset=eval_dataset)

    eval_results.update(eval_result)
    output_eval_file = os.path.join(
        output_dir, f"eval_results_{task}.txt"
    )
    if trainer.is_world_process_zero():
        with open(output_eval_file, "w") as writer:
            logging.info(f"***** Eval results {task} *****")
            for key, value in sorted(eval_result.items()):
                logging.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

    return eval_results


def test_tasks(trainer, output_dir, tasks, test_datasets, is_regression, label_list):
    """
    Write a file of predictions on the test set used for uploading to GLUE leaderboard
        TODO: make sure this works for all tasks
    """

    logging.info("At test time: total params: {:,} non zero params: {:,}".format(
        *count_nonzero_params(trainer.model)
    ))

    for test_dataset, task in zip(test_datasets, tasks):
        # Removing the `label` columns because it contains -1
        # and Trainer won't like that.
        test_dataset.remove_columns_("label")
        predictions = trainer.predict(test_dataset=test_dataset).predictions
        predictions = (np.squeeze(predictions) if is_regression
                       else np.argmax(predictions, axis=1))

        output_test_file = os.path.join(
            output_dir, f"{GLUE_NAMES_PER_TASK[task]}.tsv"
        )
        if trainer.is_world_process_zero():
            with open(output_test_file, "w") as writer:
                logging.info(f"***** Test results {task} *****")
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    if is_regression:
                        writer.write(f"{index}\t{item:3.3f}\n")
                    else:
                        item = label_list[item]
                        writer.write(f"{index}\t{item}\n")


def evaluate_language_model(trainer, eval_dataset, output_dir, metric_callback=None):
    """Evaluate language model. Returns dict with results on perplexity metric. """
    results = {}
    eval_output = trainer.evaluate(eval_dataset)

    perplexity = math.exp(eval_output["eval_loss"])
    results["perplexity"] = perplexity

    output_eval_file = os.path.join(output_dir, "eval_results_mlm.txt")
    if trainer.is_world_process_zero():
        with open(output_eval_file, "w") as writer:
            logging.info("***** Eval results *****")
            for key, value in sorted(results.items()):
                logging.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

        if metric_callback:
            tracked_eval_metrics = metric_callback.eval_metrics
            with open(os.path.join(output_dir, "eval_metrics.p"), "wb") as file:
                pickle.dump(tracked_eval_metrics, file)

    return results


def load_and_concatenate_datasets(data_args):
    """Load and concatenate multiple compatible datasets"""
    train_datasets, validation_datasets = [], []
    for name, config in zip(data_args.dataset_name, data_args.dataset_config_name):

        dataset = load_dataset(name, config)
        if "validation" not in dataset.keys():
            validation_ds = load_dataset(
                name, config,
                split=f"train[:{data_args.validation_split_percentage}%]",
            )
            train_ds = load_dataset(
                name, config,
                split=f"train[{data_args.validation_split_percentage}%:]",
            )
        else:
            validation_ds = dataset["validation"]
            train_ds = dataset["train"]

        # Some specific preprocessing to align fields on known datasets
        # extraneous fields not used in language modeling are also removed
        # after preprocessing
        if name == "wikipedia":
            train_ds.remove_columns_("title")
            validation_ds.remove_columns_("title")
        elif name == "ptb_text_only":
            train_ds.rename_column_("sentence", "text")
            validation_ds.rename_column_("sentence", "text")

        train_datasets.append(train_ds)
        validation_datasets.append(validation_ds)

    for ds_idx in range(1, len(train_datasets)):
        assert train_datasets[ds_idx].features.type == \
            train_datasets[ds_idx - 1].features.type, \
            "Features name and type must match between all datasets"

    datasets = DatasetDict()
    datasets["train"] = concatenate_datasets(train_datasets)
    datasets["validation"] = concatenate_datasets(validation_datasets)

    return datasets


def preprocess_datasets_mlm(datasets, tokenizer, data_args, column_names,
                            text_column_name):
    """Tokenize datasets and applies remaining preprocessing steps"""

    # Letting the tokenizer handle multi-threading results in poor performance
    # So if num_workers is not specified, critical to set it
    if data_args.preprocessing_num_workers is None:
        num_procs = multiprocessing.cpu_count()
    else:
        num_procs = data_args.preprocessing_num_workers

    if data_args.line_by_line:
        # When using line_by_line, we just tokenize each nonempty line.
        padding = "max_length" if data_args.pad_to_max_length else False

        def tokenize_function(examples):
            # Remove empty lines
            examples["text"] = [
                line for line in examples["text"]
                if len(line) > 0 and not line.isspace()
            ]
            return tokenizer(
                examples["text"],
                padding=padding,
                truncation=True,
                max_length=data_args.max_seq_length,
                # We use this option because DataCollatorForLanguageModeling (see below)
                # is more efficient when it receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )

        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            num_proc=num_procs,
            remove_columns=[text_column_name],
            load_from_cache_file=not data_args.overwrite_cache,
        )
    else:
        # Otherwise, we tokenize every text, then concatenate them together before
        # splitting them in smaller parts. We use `return_special_tokens_mask=True`
        # because DataCollatorForLanguageModeling (see below) is more
        # efficient when it receives the `special_tokens_mask`.
        def tokenize_function(examples):
            return tokenizer(
                examples[text_column_name], return_special_tokens_mask=True
            )

        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            num_proc=num_procs,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

        if data_args.max_seq_length is None:
            max_seq_length = tokenizer.model_max_length
            if max_seq_length > 1024:
                logging.warning(
                    f"The tokenizer picked seems to have a very large "
                    f"`model_max_length` ({tokenizer.model_max_length}). "
                    f"Picking 1024 instead. You can change that default value by "
                    f"passing --max_seq_length xxx. "
                )
                max_seq_length = 1024
        else:
            if data_args.max_seq_length > tokenizer.model_max_length:
                logging.warning(
                    f"The max_seq_length passed ({data_args.max_seq_length}) is larger "
                    f"than the maximum length for the model "
                    f"({tokenizer.model_max_length}). "
                    f"Using max_seq_length={tokenizer.model_max_length}. "
                )
            max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

        # Main data processing function that will concatenate all texts from our dataset
        # and generate chunks of max_seq_length.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported
            # it instead of this drop, you can customize this part to your needs.
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
        # You can adjust that batch_size here but a higher value might be slower
        # to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the
        # map method for more information:
        tokenized_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=num_procs,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    return tokenized_datasets


# TODO
# {ossibly refactor this function so it tokenizes the data in a default way
# without needing access to a loaded model, since it is currently
# requiring us to load a model twice for hyperparameter tuning.
# Instead we can add checks right before training to see if we need
# to tokenize differently.
def preprocess_datasets_task(datasets, tokenizer, data_args, model,
                             num_labels, label_list, is_regression):
    """Preprocess datasets for finetuning"""

    if data_args.task_name is not None:
        sentence1_key, sentence2_key = TASK_TO_KEYS[data_args.task_name]
    else:
        # Defaults, but don't hesitate to tweak to your use case.
        non_label_column_names = [
            name for name in datasets["train"].column_names if name != "label"
        ]
        if ("sentence1" in non_label_column_names
           and "sentence2" in non_label_column_names):
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # Pad later, dynamically at batch creation, to the max sequence length per batch
        padding = False

    # Some models have set the order of the labels to use, make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {
                i: label_name_to_id[label_list[i]] for i in range(num_labels)
            }
        else:
            logging.warn(
                "Your model seems to have been trained with labels, but they don't "
                "match the dataset. ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, " ,
                f"dataset labels: {list(sorted(label_list))}.",
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logging.warn(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than ",
            f"the maximum length for the model ({tokenizer.model_max_length}). ",
            f"Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(
            *args, padding=padding, max_length=max_seq_length, truncation=True
        )

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [label_to_id[label] for label in examples["label"]]
        return result

    tokenized_datasets = datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=not data_args.overwrite_cache
    )

    return tokenized_datasets


def init_datasets_mlm(data_args):
    """
    Initialize datasets.
    Returns a regular version of dataset and a tokenized version if exists.

    Get the datasets: you can either provide your own CSV/JSON/TXT training
    and evaluation files (see below) or just provide the name of one of the public
    datasets available on the hub at https://huggingface.co/datasets/
    (the dataset will be downloaded automatically from the datasets Hub

    For CSV/JSON files, this script will use the column called 'text' or the first
    column. You can easily tweak this behavior (see below)

    In distributed training, the load_dataset function guarantee that only one local
    process can concurrently download the dataset.

    See more about loading any type of standard or custom dataset
    (from files, python dict, pandas DataFrame, etc) at
    https://huggingface.co/docs/datasets/loading_datasets.html.
    """

    datasets, tokenized_datasets, dataset_path = None, None, None

    if data_args.dataset_name is not None:
        # Load from dataset instead
        if not(isinstance(data_args.dataset_name, tuple)):
            # account for when a single dataset is passed as argument
            data_args.dataset_name = (data_args.dataset_name, )
        if not (isinstance(data_args.dataset_config_name, tuple)):
            data_args.dataset_config_name = (data_args.dataset_config_name, )

        assert len(data_args.dataset_name) == len(data_args.dataset_config_name), \
            ("If using more than one dataset, length of dataset_name and "
             "dataset_config_name must match")

        # Verifies if dataset is already saved
        dataset_folder = hash_dataset_folder_name(data_args)
        dataset_path = os.path.join(
            os.path.abspath(data_args.tokenized_data_cache_dir),
            str(dataset_folder)
        )
        logging.info(f"Tokenized dataset cache folder: {dataset_path}")

        if os.path.exists(dataset_path) and data_args.reuse_tokenized_data:
            logging.info(f"Loading cached tokenized data ...")
            tokenized_datasets = load_from_disk(dataset_path)
        else:
            datasets = load_and_concatenate_datasets(data_args)

    else:
        # If dataset not available in Hub, look for train and validation files.
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = data_args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        datasets = load_dataset(extension, data_files=data_files)

    return datasets, tokenized_datasets, dataset_path


def init_datasets_task(data_args, training_args):
    """
    Get the datasets for finetuning on tasks. Returns dataset.

    You can either provide your own CSV/JSON training and evaluation files (see below)
    or specify a GLUE benchmark task (the dataset will be downloaded automatically
    from the datasets Hub).

    For CSV/JSON files, this script will use as labels the column called 'label' and as
    pair of sentences the sentences in columns called 'sentence1' and 'sentence2' if
    such column exists or the first two columns not named label if at least two columns
    are provided.

    If the CSVs/JSONs contain only one non-label column, the script does single sentence
    classification on this single column. You can easily tweak this behavior (see below)

    In distributed training, the load_dataset function guarantee that only one local
    process can concurrently download the dataset.

    See more about loading any type of standard or custom dataset at
    https://huggingface.co/docs/datasets/loading_datasets.html.

    TODO: implement recovering tokenized version
    """

    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset("glue", data_args.task_name)
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {
            "train": data_args.train_file, "validation": data_args.validation_file
        }

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert test_extension == train_extension, \
                    "`test_file` should have the same extension (csv or json) " \
                    + "as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError(
                    "Need either a GLUE task or a test file for `do_predict`."
                )

        for key in data_files.keys():
            logging.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            datasets = load_dataset("csv", data_files=data_files)
        else:
            # Loading a dataset from local json files
            datasets = load_dataset("json", data_files=data_files)

    return datasets


def get_labels(datasets, data_args):
    label_list = None
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = (
            datasets["train"].features["label"].dtype in ["float32", "float64"]
        )
        if is_regression:
            num_labels = 1
        else:
            label_list = datasets["train"].unique("label")
            label_list.sort()  # sort it for determinism
            num_labels = len(label_list)

    return is_regression, label_list, num_labels


def init_config(model_args, extra_config_kwargs=None):
    """
    Distributed training:
    The .from_pretrained methods guarantee that only one local process can
    concurrently download model & vocab.
    """
    config_kwargs = dict(
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if extra_config_kwargs is not None:
        config_kwargs.update(extra_config_kwargs)

    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, **config_kwargs
        )
    else:
        config = CONFIG_MAPPING[model_args.model_type](**model_args.config_kwargs)
        logging.warning("You are instantiating a new config instance from scratch.")

    return config


def init_tokenizer(model_args):
    """
    Distributed training:
    The .from_pretrained methods guarantee that only one local process can
    concurrently download model & vocab.
    """
    tokenizer_kwargs = dict(
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, **tokenizer_kwargs
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, **tokenizer_kwargs
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported "
            "by this script. You can do it from another script, save it, and load it "
            "from here, using --tokenizer_name."
        )

    return tokenizer


def init_model(model_args, config, tokenizer, finetuning=False):
    """"
    Initialize a model for pretraining or finetuning
    """

    # Load model
    if model_args.model_name_or_path is not None:
        model_kwargs = dict(
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        if finetuning:
            logging.info("Loading a pretrained model for finetuning")
            model = AutoModelForSequenceClassification.from_pretrained(
                model_args.model_name_or_path, **model_kwargs
            )
        else:
            logging.info("Loading a pretrained model to continue pretraining")
            model = AutoModelForMaskedLM.from_pretrained(
                model_args.model_name_or_path, **model_kwargs
            )
            model.resize_token_embeddings(len(tokenizer))
    else:
        if finetuning:
            raise ValueError(
                "Finetuning models must be loaded from pretrained models."
            )
        else:
            logging.info("Pretraining new model from scratch")
            model = AutoModelForMaskedLM.from_config(config)
            model.resize_token_embeddings(len(tokenizer))

    logging.info(f"Initialized model: {model}")
    return model


def toggle_drop_last_wrapper(method):
    """
    Return a function that turns drop_last off before it is called. Used for
    ensuring trainer.args.dataloader_drop_last is False during evaluation
    steps. After the method is called, dataloader_drop_last is switched back
    to whatever it was set to initially.
    """
    def toggle_method(*args, **kwargs):
        was_drop_last = method.__self__.args.dataloader_drop_last  # initial drop_last
        method.__self__.args.dataloader_drop_last = False  # turn drop_last off
        result = method(*args, **kwargs)  # call method with drop_last off
        method.__self__.args.dataloader_drop_last = was_drop_last  # restore drop_last
        return result

    return toggle_method


def format_eval_results(eval_results, run, task_name):

    ignore_keys = ["steps", "epoch"]
    new_eval_results = {}
    for key in eval_results.keys():
        if key in ignore_keys:
            new_eval_results[key] = eval_results[key]
        else:
            new_key = task_name + f"run{run}_" + key
            new_eval_results[new_key] = eval_results[key]

    return new_eval_results

#####
# Code to make sure training / finetuning / hp optimization runs safely
#####


def check_for_callback(model_args, class_of_callback):

    has_callback = False
    callback_instance = None
    for callback_idx in range(len(model_args.trainer_callbacks)):
        if isinstance(model_args.trainer_callbacks[callback_idx], class_of_callback):
            has_callback = True
            callback_instance = model_args.trainer_callbacks[callback_idx]

    return has_callback, callback_instance


def check_sparsity_callback(model, model_args):

    is_sparse = False
    for module in model.modules():
        if isinstance(module, SparseWeightsBase):
            is_sparse = True
            break

    if is_sparse:
        has_rezero = check_for_callback(model_args, RezeroWeightsCallback)
        assert has_rezero[0], "Finetuning sparse models without rezeroing weights"
        " is prohibited"


def check_eval_and_max_steps(training_args, train_dataset):
    """
    If you're supposed to load best model at end, but evaluate() never gets
    called because eval_steps > number of training steps taken, you'll get
    an error at the end of training. If this is the case, set eval_steps
    equal to max_steps so it gets called at least once.
    """
    if training_args.load_best_model_at_end:
        if training_args.max_steps == -1:
            num_examples = training_args.num_train_epochs * len(train_dataset)
            max_steps = num_examples // training_args.per_device_train_batch_size
        else:
            max_steps = training_args.max_steps
        if max_steps < training_args.eval_steps:
            logging.warning(
                f"max_steps({max_steps}) < "
                f"eval_steps({training_args.eval_steps}) "
                "To avoid issues, setting eval steps equal to max_steps"
            )
            training_args.eval_steps = max_steps

    return training_args


def check_hp_compute_objective(model_args,
                               task_name,
                               training_args):
    """
    When hyperparameter tuning, you need to specify an objective, like
    eval_accuracy, and also if it should be minimized or maximized. It is easy
    to make a mistake because you copied a config and changed the task, but
    the objective no longer applies to this task. Or, you changed the
    objective but forget to switch minimize / maximize. This function checks
    for both types of mistakes and issues a warning before correcting it.
    """

    hp_compute_objective = getattr(model_args, "hp_compute_objective", None)
    if hp_compute_objective is not None:
        direction, objective = model_args.hp_compute_objective
        if "eval_loss" in objective:
            if direction != "minimize":
                logging.warning(
                    "You are asking hp search to find parameters"
                    "that MAXIMIZE loss instead of MINIMIZING it."
                    "Setting this to minimize"
                )
                # Can't modify tuples, so convert to list and then back
                hp_compute_objective = list(model_args.hp_compute_objective)
                hp_compute_objective[0] = "minimize"
                model_args.hp_compute_objective = tuple(hp_compute_objective)
        else:
            allowed_metrics = get_allowed_metrics(training_args, task_name)
            if objective not in allowed_metrics:
                logging.warning(
                    "Warning, code will break when you try to tune"
                    "hyperparameters on this task because"
                    "hp_compute_objective is incorrect. Setting it to"
                    "first reporting metric"
                )
                hp_compute_objective = list(model_args.hp_compute_objective)
                hp_compute_objective[1] = REPORTING_METRICS_PER_TASK[task_name][0]
                hp_compute_objective[0] = "maximize"
                model_args.hp_compute_objective = tuple(hp_compute_objective)

    return model_args


def get_allowed_metrics(training_args, task_name):

    allowed_metrics = list(REPORTING_METRICS_PER_TASK[task_name])
    allowed_metrics.append("eval_loss")

    # In the special case of mnli, you have multiple validation sets
    # A prefix (m, or mm) is used to distinguish them. In this case,
    # overwrite allowed_metrics to use the prefixes
    if training_args:
        if "eval_prefixes" in training_args.trainer_mixin_args:
            allowed_metrics = RAW_REPORTING_METRICS_PER_TASK[task_name]
            allowed_metrics.append("loss")
            if task_name != "mnli":
                raise NotImplementedError
            else:
                prefixed_allowed_metrics = []
                prefixes = training_args.trainer_mixin_args["eval_prefixes"]
                for metric in allowed_metrics:
                    for prefix in prefixes:
                        prefixed_allowed_metrics.append("_".join(
                            [prefix, metric]))

                allowed_metrics = prefixed_allowed_metrics

    return allowed_metrics


def check_metric_direction(metric, greater_is_better):

    if greater_is_better is None:
        greater_is_better = False
    if "loss" in metric:
        if greater_is_better:
            logging.warning(
                "Greater is better is set to True with eval_loss as "
                "metric_for_best_model. Flipping greater is better to"
                "False, since we want small loss"
            )
        return False
    else:
        if not greater_is_better:
            logging.warning(
                "Greater is better is set to False with non-loss "
                f"metric {metric}. Setting greater is better to True"
            )
        return True


def check_metric_is_allowed(metric, allowed_metrics, task_name=None):

    if metric not in allowed_metrics:
        if task_name == "mnli":
            # Use the mismatched val set for mnli to avoid overfitting
            new_metric = allowed_metrics[1]
        else:
            new_metric = allowed_metrics[0]
        logging.warning(
            "Warning, code will break because the current metric for best model"
            f" ({metric}) is not being tracked."
            "Defaulting metric_for_best_model to {new_metric}"
        )
        return new_metric
    else:
        return metric


def check_best_metric(training_args, task_name, metric=None):
    """
    Runs can easily break if load_best_model_at_end because you
    specified a metric for a diferent task. You can get all the way through
    training and have it break. This will flip best_metric to eval_loss in
    that case. It also checks to make sure greater_is_better is set properly.
    """

    if metric is None:
        metric = training_args.metric_for_best_model
    allowed_metrics = get_allowed_metrics(training_args, task_name)
    metric = check_metric_is_allowed(metric,
                                     allowed_metrics,
                                     task_name)
    greater_is_better = training_args.greater_is_better
    greater_is_better = check_metric_direction(
        metric, greater_is_better
    )

    print("metric configuration after checks: "
          f"{metric}, {greater_is_better}")
    training_args.greater_is_beter = greater_is_better
    training_args.metric_for_best_model = metric

    return training_args


def check_mnli(model_args, task_name):
    """
    There are multiple way to handle mnli which has multiple eva;l sets
    However, the recommended approach is to simply use multi_eval_sets
    callback. This warns you if you are not doing that.
    """

    if task_name == "mnli":
        if "MultiEvalSetTrainer" in str(model_args.trainer_class):
            logging.info("Using recommended multi eval set approach for mnli")
        else:
            logging.warn(
                "You are training on mnli without multi eval sets!"
                "This is strongly discouraged and QA is not guaranteed!"
            )


def check_rm_checkpoints(training_args, model_args):
    # If pretraining, you usually want to save checkpoints.
    if not model_args.finetuning:
        if training_args.rm_checkpoints:
            logging.warning(
                "Warning, rm_checkpoints is set to true for this pretraining"
                "experiment. That means a single model will be saved and all "
                "checkpoint subdirectories will be deleted."
            )

    else:
        if not training_args.rm_checkpoints:
            logging.warning(
                "Warning, rm_checkpoints is set to false for this finetuning"
                "experiment. That means all model checkpoints will be saved, "
                "which takes up a lot of space"
            )


def evaluate_task_handler(trainer,
                          data_args,
                          model_args,
                          training_args,
                          eval_dataset,
                          tokenized_datasets):
    """
    Handle the last evaluation. If you've been evaluating throughout the
    training process, evaluate again if steps % eval steps is jagged.

    This determines if you should evaluate on a single task. If mnli, assume
    multi_eval_sets mixin is in use which handles mismatched set for you.
    """
    logging.info("*** Evaluate ***")

    eval_results = {}
    tracked_metrics, metric_callback = check_for_callback(model_args, TrackEvalMetrics)
    task = data_args.task_name

    if tracked_metrics:

        tracked_eval_metrics = metric_callback.eval_metrics
        tracked_eval_metrics["steps"] = metric_callback.steps
        offset = training_args.max_steps % training_args.eval_steps

        # If max_steps % eval_steps is not 0, make sure to eval one last time
        if offset != 0 and not training_args.load_best_model_at_end:

            print(f"Evaluating again because offset was {offset}")

            eval_results = evaluate_task(
                trainer, training_args.output_dir, task, eval_dataset
            )
            metric_callback.eval_metrics["steps"][-1] -= offset

        eval_results = tracked_eval_metrics

    else:

        eval_results = evaluate_task(
            trainer, training_args.output_dir, task, eval_dataset
        )

    return eval_results


def init_trainer(
    tokenizer,
    data_collator,
    training_args,
    train_dataset,
    eval_dataset,
    model=None,
    trainer_callbacks=None,
    finetuning=False,
    task_name=None,
    is_regression=False,
    trainer_class=Trainer,
    model_init=None,
):
    """Initialize Trainer, main class that controls the experiment"""
    if trainer_callbacks is not None:
        for cb in trainer_callbacks:
            assert isinstance(cb, TrainerCallback), \
                "Trainer callbacks must be an instance of TrainerCallback"

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=trainer_callbacks,
        model_init=model_init,
    )

    # Add specific metrics for finetuning task
    if finetuning:
        compute_metrics = partial(
            compute_metrics_task,
            task_name=task_name,
            is_regression=is_regression,
            output_dir=training_args.output_dir
        )
        trainer_kwargs.update(compute_metrics=compute_metrics)

    trainer = trainer_class(**trainer_kwargs)

    # Issue: labels get set to -100 due to drop_last.
    # Fix: override the evaluate and predict methods.
    # The previous fix covered cases when WE call trainer.{evaluate, predict}.
    # This fix should cover all cases, including any time HF calls these methods.
    trainer.evaluate = toggle_drop_last_wrapper(trainer.evaluate)
    trainer.predict = toggle_drop_last_wrapper(trainer.predict)

    return trainer


def compute_metrics_task(ep: EvalPrediction, metric=None,
                         task_name=None, is_regression=None,
                         output_dir=None):
    """
    You can define your custom compute_metrics function. It takes an
    `EvalPrediction` object (a namedtuple with a predictions and label_ids
    field) and has to return a dictionary string to float.
    """
    preds = (ep.predictions[0] if isinstance(ep.predictions, tuple) else ep.predictions)
    preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)

    # -100 labels can come up when drop_last batch setting gets set to true during
    # evaluation. That is fixed, so any -100 labels should not pass silently.
    assert -100 not in ep.label_ids, "Unknown source of -100 labels"

    if not is_regression:
        logging.info(f"Label distribution for {task_name}")
        logging.info(f"Predictions: {Counter(preds).most_common()}")
        logging.info(f"Labels: {Counter(ep.label_ids).most_common()}")

    if task_name is not None:
        if task_name == "cola":
            result = {"matthews_correlation": matthews_corrcoef(ep.label_ids, preds)}
        elif task_name == "stsb":
            result = pearson_and_spearman(preds, ep.label_ids)
        elif task_name in ["mrpc", "qqp"]:
            result = acc_and_f1(preds, ep.label_ids)
        elif task_name in ["sst2", "mnli", "mnli_matched", "mnli-mm", "mnli_mismatched",
                           "qnli", "rte", "wnli", "hans"]:
            result = {"accuracy": simple_accuracy(preds, ep.label_ids)}
        # Consolidate if more than one metric
        if len(result) > 1:
            combined_score = np.mean(list(result.values())).item()
            result[task_name + "_combined_score"] = combined_score
        return result
    elif is_regression:
        return {"mse": ((preds - ep.label_ids) ** 2).mean().item()}
    else:
        return {"accuracy": (preds == ep.label_ids).astype(np.float32).mean().item()}


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "accuracy": acc,
        "f1": f1,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
    }


class TaskResults():
    """
    Collects and reports results for each finetuning task
    """

    reporting_metrics_per_task = REPORTING_METRICS_PER_TASK

    def __init__(self, task_name, training_args):
        self.task_name = task_name
        self.reporting_metrics = self.reporting_metrics_per_task[self.task_name]
        self.allowed_metrics = get_allowed_metrics(training_args, task_name)
        self.all_results = []
        self._results = None
        self.training_args = training_args
        self.load_best_model_at_end = \
            self.training_args.load_best_model_at_end
        self.best_metric_key = self.training_args.metric_for_best_model
        # If early stopping, need to track which step best results were at
        # If not, -1 corresponds to end of training
        self.best_idx_per_run = []  # One index for each run

    def __len__(self):
        return len(self.all_results)

    def get_formatted_results(self):
        """
        Format eval_results to include task_name and run_idx for clarity when analyzing
        multiple runs on multiple finetuning tasks.
        """
        self.all_fmt_results = []
        for i in range(len(self.all_results)):
            fmt_result = format_eval_results(self.all_results[i], i, self.task_name)
            self.all_fmt_results.append(fmt_result)

    def append(self, results):
        """
        Add results to all_results. Since results is a dict that can have values that
        are either numbers or lists of numbers, pack all values in a list so they can
        be processed more easily in subsequent steps.
        """
        self.all_results.append({})
        for key in results.keys():
            if not isinstance(results[key], list):
                self.all_results[-1][key] = [results[key]]
            else:
                self.all_results[-1][key] = results[key]

    def reduce_metrics(self, reduction="mean"):
        """
        Get average or max over runs. Handles two cases:
            1) If using early stopping, find the index where metric_for_best_model
               is highest in each run. Get the max or mean of these.
            2) If not using early stopping, get the last entry in each run, and take
               the max or mean of these. This entry corresponds to the end of training.

        Note that if you're not using TrackEvalMetrics callback, results dictionaries
        are formatted so the values are lists fo length 1. Cases (1) and (2) above
        result in the same behavior in this case, since there is a single entry to
        reduce over in each run.
        """
        # all_results[run_idx][metric] is a number if not tracking eval metrics,
        # or a list with a number for each time evaluate() is called.
        # aggregated_results[metric] is a list of metric values, one for each run
        aggregated_results = defaultdict(list)
        load_best = getattr(self, "load_best_model_at_end", None)
        stop_early = getattr(self, "early_stopping", None)
        load_best_or_stop_early = load_best or stop_early
        # Loop over runs on the same task
        for results in self.all_results:
            if load_best_or_stop_early:
                # Within a run, the step where best results were achieved
                # Note, metrics defined in finetuning_constants are all better
                # when higher, so no need to worry about argmin.
                best_metric_best_idx = np.argmax(results[self.best_metric_key])
            else:
                # If not load best at end, just get the last step
                best_metric_best_idx = -1

            self.best_idx_per_run.append(best_metric_best_idx)
            for metric, values in results.items():
                aggregated_results[metric].append(values[best_metric_best_idx])

        # Average across runs
        if reduction == "mean":
            self._results = {k: np.average(v) for k, v in aggregated_results.items()}

        # Max across runs
        elif reduction == "max":
            # Which run has the best results
            argmax_run = np.argmax(aggregated_results[self.reporting_metrics[-1]])
            # argmax_run = np.argmax(aggregated_results[self.best_metric_key])
            # Which step in the run has best results
            argmax_step = self.best_idx_per_run[argmax_run]
            self._results = {}
            for k, v in self.all_results[argmax_run].items():
                self._results[k] = v[argmax_step]

    @property
    def results(self):
        if self._results is not None:
            return self._results
        elif len(self.all_results) > 0:
            return self.all_results[0]
        else:
            raise AttributeError(f"Results not available for {self.task_name}")

    def consolidate(self):
        return np.average([self.results[m] for m in self.reporting_metrics])

    def to_string(self):
        results_to_string = [
            f"{self.results[m]*100:.2f}" for m in self.reporting_metrics
        ]
        return "/".join(results_to_string)

    def get_model_with_best_max(self, metric=None):
        """
        Utility added to get predictions of the best model at the end of
        run_finetuning_multiple_tasks. For now this is assuming
        load_best_model_at_end is True
        """

        if not metric:
            metric = self.best_metric_key

        if metric not in self.all_results[0]:
            metric = self.reporting_metrics[0]

        print(f"Selecting best model based on {metric}")
        greater_is_better = True
        op = max
        if "loss" in metric:
            greater_is_better = False
            op = min

        # Best index in each run
        bests = [
            op(self.all_results[i][metric])
            for i in range(len(self.all_results))
        ]

        # Run with the best, best idx
        if greater_is_better:
            best_model_idx = np.argmax(bests)
        else:
            best_model_idx = np.argmin(bests)

        return best_model_idx


def hash_dataset_folder_name(data_args):
    """
    Creates a hashed name for the dataset folder comprised of the dataset_name and
    dataset_name_config. As well, the following data_args are included unless their
    default values are used.
        - max_seq_length (default None)

    More arguments can be added to the hashed name as needed.
    """
    defaults = dict(
        max_seq_length=None,
    )

    dataset_folder = "-".join([
        f"{name}_{config}" for name, config in
        zip(data_args.dataset_name, data_args.dataset_config_name)
    ])

    for arg, default in defaults.items():
        if getattr(data_args, arg) != default:
            non_default = getattr(data_args, arg)
            dataset_folder += f" ({arg}={non_default})"

    hashed_folder_name = blake2b(dataset_folder.encode(), digest_size=20).hexdigest()
    logging.info("Hashing dataset folder name "
                 f"'{dataset_folder}' to '{hashed_folder_name}'")
    return hashed_folder_name


def run_hyperparameter_search(
    model_args,
    config,
    tokenizer,
    data_collator,
    training_args,
    train_dataset,
    eval_dataset,
):
    """
    Run hyperparameter search using Ray Tune
    Not tested when using multiple instances with torch.distributed.launch
    """

    training_args.load_best_model_at_end = True
    training_args.disable_tqdm = True  # competes with ray output
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

    # Specify how to re-init model each training run.
    def model_init():

        # Our custom model mapping made for sparse models must be imported here
        # as ray uses an independently imported version of transformers which
        # doesn't have access to this updated mapping.
        from models import MODEL_FOR_MASKED_LM_MAPPING as CUSTOM_MASKED_LM_MAPPING
        from models import CONFIG_MAPPING as CUSTOM_CONFIG_MAPPING

        # For now, we'll only load new models from scratch.
        assert model_args.model_name_or_path is None, \
            "HP search with saved models not supported."
        logging.info("Pretraining new model from scratch")

        # Instantiate model; possibly one of our custom sparse models.
        config_cls = CUSTOM_CONFIG_MAPPING[config.model_type]
        model_for_lm_cls = CUSTOM_MASKED_LM_MAPPING[config_cls]
        model = model_for_lm_cls(config)
        model.resize_token_embeddings(len(tokenizer))
        return model

    trainer = init_trainer(
        tokenizer=tokenizer,
        data_collator=data_collator,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=hp_eval_dataset,
        trainer_class=model_args.trainer_class,
        trainer_callbacks=model_args.trainer_callbacks or None,
        model_init=model_init,
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


def compute_objective(metrics, objective):
    return metrics[objective]


def check_if_current_hp_best(old_file, model_args, best_run):
    """
    If you run multiple finetuning experiments, you want one file with
    the best params / results. If the current run beats the old best,
    overwrite the best_run_results file. This function returns a bool,
    indicating if you should overwrite, or not.
    """

    if not best_run:
        return False
    # If there is no current file, write a new file
    if not os.path.exists(old_file):
        return True

    # Read the file with current best results
    with open(old_file, "r") as f:
        data = f.read()
        line_split = data.split("\n")

    # Search though the file to get the best score
    previous_best = None
    for line in line_split:
        if model_args.hp_compute_objective[1] in line:
            previous_best = float(line.split("=")[-1])

    # To compare new scores with current scores, need to decide
    # if greater is better.
    if model_args.hp_compute_objective[0] == "maximize":
        return best_run.objective > previous_best
    else:
        return best_run.objective < previous_best


def collate_hp_csvs(report_dir):

    files = os.listdir(report_dir)
    subdirs = []
    for file in files:
        full_path = os.path.join(report_dir, file)
        if os.path.isdir(full_path):
            subdirs.append(full_path)

    dfs = []
    for subdir in subdirs:
        df = pd.read_csv(os.path.join(subdir, "progress.csv"))
        dfs.append(df)

    big_df = pd.concat(dfs)
    big_df.to_csv(os.path.join(report_dir, "all_progress.csv"))


def update_run_number(training_args, run_idx):
    """
    Simple util so that when you are finetuning one task for multiple runs,
    we update the output directory for each run.
    """
    if run_idx is None:
        return training_args

    if run_idx == 0:
        new_dir = os.path.join(training_args.output_dir, "run_0")
    else:
        dirname, basename = os.path.split(training_args.output_dir)
        new_dir = os.path.join(dirname, f"run_{run_idx}")
    training_args.output_dir = new_dir
    return training_args


def get_best_run_and_link_best_predictions(training_args,
                                           task_results,
                                           task_name):
    """
    When training for multiple runs on one task, find the best run so that
    you can delete the other directories. If you are predicting on the test
    set, then create a symlink between {task_name}_best.tsv and the test set
    predictions of the run with the best eval scores. (e.g. CoLA_best.tsv).
    """

    # get the filename with predictions from the best model
    best_run = task_results.get_model_with_best_max()
    task_path = os.path.dirname(training_args.output_dir)
    best_run_path = os.path.join(task_path, f"run_{best_run}")

    pred_files = [GLUE_NAMES_PER_TASK[task_name] + ".tsv"]
    if task_name == "mnli":
        pred_files.append(GLUE_NAMES_PER_TASK[task_name + "-mm"] + ".tsv")

    best_run_predictions = [os.path.join(best_run_path, pred_file)
                            for pred_file in pred_files]

    # You want to find best_run in all cases. You want to link
    # to a set of predictions only when do_predict is on.
    if training_args.do_predict:

        # link task_best.tsv
        # If previous symlink exists, just delete and recreate
        link_file_names = [GLUE_NAMES_PER_TASK[task_name] + "_best.tsv"]
        if task_name == "mnli":
            link_file_names.append(
                GLUE_NAMES_PER_TASK[task_name + "-mm"] + "_best.tsv")

        for idx, link_file_name in enumerate(link_file_names):
            link_file_path = os.path.join(task_path, link_file_name)
            if os.path.exists(link_file_path):
                os.remove(link_file_path)
            os.symlink(best_run_predictions[idx], link_file_path)
            logging.info(f"best run predictions for {task_name} saved to "
                         f"{link_file_path}")

    return str(best_run)


def rm_prefixed_subdirs(base_dir, prefix, skip=""):
    """
    Remove unnecessary directories at the end of training
    """
    logging.info(f"Removing {prefix}* dirctories in {base_dir}")
    for subdir in os.listdir(base_dir):
        if subdir.startswith(prefix):
            subdir_path = os.path.join(base_dir, subdir)
            if subdir == skip:
                logging.info(f"Not deleting {subdir}")
            else:
                shutil.rmtree(subdir_path)
