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

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch
from transformers import MODEL_FOR_MASKED_LM_MAPPING, Trainer, TrainingArguments

from run_utils import TASK_TO_KEYS, compute_objective_eval_loss

MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class CustomTrainingArguments(TrainingArguments):
    """
    Additional arguments to HF TrainingArguments
    """
    num_runs: int = field(
        default=1,
        metadata={
            "help": "How many runs per task. Currently only used for finetuning."
        },
    )
    trainer_mixin_args: Dict = field(
        default_factory=dict,
        metadata={
            "help": "Extra arguments to be passed to Trainer. Can be accessed "
                    "for addition arguments when trainer mixins are used."
        }
    )
    hp_space: Callable = field(
        default=lambda: {},
        metadata={
            "help": "Hyperparameters to search for in the model"
        }
    )
    hp_num_trials: int = field(
        default=0,
        metadata={
            "help": "How many trials to run during hyperparameter search"
        }
    )
    hp_validation_dataset_pct: float = field(
        default=0.05,
        metadata={
            "help": "Percentage of the validation dataset to be used in hp search"
        }
    )
    hp_compute_objective: Callable = field(
        default=compute_objective_eval_loss,
        metadata={
            "help": "Defines the objective function be used in hyperparameter search"
        }
    )
    hp_extra_kwargs: Dict = field(
        default_factory=dict,
        metadata={
            "help": "Dictionary with extra parameters to be passed to "
                    "`trainer.hyperparameter_search`. Includse arguments to `tune.run`"
        }
    )
    hp_resources_per_trial: Dict = field(
        default_factory=lambda: dict(
            cpu=os.cpu_count() / torch.cuda.device_count() - 1,
            gpu=1
        ),
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune,
    or train from scratch.
    """
    finetuning: bool = field(
        default=False,
        metadata={
            "help": "Whether to finetune the model for downstream tasks. If false, "
                    "will attempt to pretrain a masked language model instead."
        },
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
            "Config and tokenizers will also be determined from the model_name_or_path."
        },
    )
    model_type: Optional[str] = field(
        default="bert",
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
                    ", ".join(MODEL_TYPES)
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        }
    )
    config_kwargs: Dict = field(
        default_factory=dict,
        metadata={
            "help": "Keyword arguments to pass to model config constructor."
        }
    )
    tokenizer_name: Optional[str] = field(
        default="bert-base-cased",
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        }
    )
    cache_dir: Optional[str] = field(
        default="/mnt/efs/results/pretrained-models/huggingface",
        metadata={
            "help": "Where do you want to store the pretrained models downloaded "
                    "from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the "
                    "tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, "
                    "tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running "
                    "`transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    task_hyperparams: Dict = field(
        default_factory=dict,
        metadata={
            "help": "Allow user to define custom training arguments per task."
        },
    )
    trainer_callbacks: List = field(
        default_factory=list,
        metadata={
            "help": "List of callbacks to trainer"
        },
    )
    trainer_class: Callable = field(
        default=Trainer,
        metadata={
            "help": "Trainer class"
        }
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for
    training and eval.
    """

    tokenized_data_cache_dir: str = field(
        default="/mnt/efs/results/preprocessed-datasets/text",
        metadata={"help": "Directory to save tokenized datasets."}
    )
    reuse_tokenized_data: bool = field(
        default=False,
        metadata={
            "help": "Whether to look for a compatible tokenized version of the dataset "
                    "Contrary to huggingface caching approach, that takes into account "
                    "the entire preprocessing pipeline to calculate a fingerprint, this"
                    "only takes into account the dataset names, config names and "
                    "their respective order."
        }
    )
    save_tokenized_data: bool = field(
        default=False,
        metadata={
            "help": "Whether to save a copy of the tokenized data in the location "
                    "defined in tokenized_data_cache_dir. This is in addition to "
                    "HF/datasets caching process. If this is True, consider "
                    "turning off caching in the datasets lib to avoid redundancy. "
        }
    )
    dataset_name: Optional[str] = field(
        default="wikitext",
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default="wikitext-2-raw-v1",
        metadata={
            "help": "The configuration name of the dataset to use "
                    "(via the datasets library)."
        }
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the "
                    "perplexity on (a text file)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in "
                    "case there's no validation split. "
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. "
                    "Sequences longer than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={
            "help": "Whether distinct lines of text in the dataset are to be "
                    "handled as distinct sequences."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum "
            "length in the batch."
        },
    )
    data_collator: str = field(
        default="DataCollatorForLanguageModeling",
        metadata={
            "help": "Which data collator to use, define how masking is applied."
        },
    )
    task_names: List[str] = field(
        default_factory=list,
        metadata={
            "help": "Allows running several tasks at once. "
                    f"Available tasks: {', '.join(TASK_TO_KEYS.keys())}"
        },
    )
    task_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the task to train on."
                    "Option available for backwards compatibility with HF run script."
                    f"Available tasks: {', '.join(TASK_TO_KEYS.keys())}"
        },
    )
    override_finetuning_results: bool = field(
        default=False,
        metadata={
            "help": "Whether to create a new results file. If set to False, will only"
                    "attempt to update existing entries"
        },
    )

    def __post_init__(self):
        """Input validation"""

        # Handle string tasks, backwards compatibility and GLUE
        if self.task_name and self.task_names:
            raise ValueError(
                "Define either a single task_name or multiple tasks using task_names"
            )
        if self.task_name:
            if self.task_name.lower() == "glue":
                self.task_names = list(TASK_TO_KEYS.keys())
            else:
                self.task_names = [self.task_name]
            self.task_name = None

        # For finetuning, has to define one or more tasks among available tasks
        if self.task_names:
            # Validates that all tasks exists
            self.task_names = [t.lower() for t in self.task_names]
            for task_name in self.task_names:
                # Checks if it is a valid task
                if task_name not in TASK_TO_KEYS.keys():
                    raise ValueError(f"Unknown task {task_name}, you should pick one in"
                                     ": " + ",".join(TASK_TO_KEYS.keys()))

        # If no task is set, validates if a dataset is given
        elif (self.dataset_name is None and self.train_file is None
              and self.validation_file is None):
            raise ValueError(
                "Need either a GLUE task, a dataset name or a training/validation file."
            )

        # If a train file is given, verify if extensions are compatible
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], \
                    "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], \
                    "`validation_file` should be a csv, a json or a txt file."
