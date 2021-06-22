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

REPORTING_METRICS_PER_TASK = {
    "cola": ["eval_matthews_correlation"],
    "mnli": ["eval_accuracy", "mm_eval_accuracy"],
    "mrpc": ["eval_f1", "eval_accuracy"],
    "qnli": ["eval_accuracy"],
    "qqp": ["eval_accuracy", "eval_f1"],
    "rte": ["eval_accuracy"],
    "sst2": ["eval_accuracy"],
    "stsb": ["eval_pearson", "eval_spearmanr"],
    "wnli": ["eval_accuracy"]
}

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