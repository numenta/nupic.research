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
Constants used in various places
"""

TASK_NAMES = [
    "cola",
    "mnli",
    "mrpc",
    "qnli",
    "qqp",
    "rte",
    "sst2",
    "stsb",
    "wnli"
]

# Names of files with test set predictions for uplodaing to glue
GLUE_NAMES_PER_TASK = {
    "cola": "CoLA",
    "sst2": "SST-2",
    "mrpc": "MRPC",
    "stsb": "STS-B",
    "qqp": "QQP",
    "mnli": "MNLI-M",
    "mnli-mm": "MNLI-MM",
    "qnli": "QNLI",
    "rte": "RTE",
    "wnli": "WNLI",
    "": "AX"  # diagnostic set, not yet implemented
}

# Metrics in each task prior to HuggingFace prepending "eval_"
RAW_REPORTING_METRICS_PER_TASK = {
    "cola": ["matthews_correlation"],
    "mnli": ["accuracy"],
    "mrpc": ["f1", "accuracy"],
    "qnli": ["accuracy"],
    "qqp": ["accuracy", "f1"],
    "rte": ["accuracy"],
    "sst2": ["accuracy"],
    "stsb": ["pearson", "spearmanr"],
    "wnli": ["accuracy"]
}

REPORTING_METRICS_PER_TASK = {
    "cola": ["eval_matthews_correlation"],
    "mnli": ["eval_accuracy", "eval_mm_accuracy"],
    "mrpc": ["eval_f1", "eval_accuracy"],
    "qnli": ["eval_accuracy"],
    "qqp": ["eval_accuracy", "eval_f1"],
    "rte": ["eval_accuracy"],
    "sst2": ["eval_accuracy"],
    "stsb": ["eval_pearson", "eval_spearmanr"],
    "wnli": ["eval_accuracy"]
}

ALL_REPORTING_METRICS = [
    "eval_matthews_correlation",
    "eval_accuracy",
    "eval_mm_accuracy",
    "eval_f1",
    "eval_pearson",
    "eval_spearmanr",
    "eval_loss"
]

# These are approximate and taken from the table in the paper
# https://arxiv.org/pdf/1804.07461.pdf
TRAIN_SIZES_PER_TASK = {
    "cola": 8_500,
    "mnli": 393_000,
    "mrpc": 3_700,
    "qnli": 105_000,
    "qqp": 364_000,
    "rte": 2_500,
    "sst2": 67_000,
    "stsb": 7_000,
    "wnli": 634,
}
