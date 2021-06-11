from .default_base import CONFIGS as DEFAULT_BASE
from .sparse_resnets import CONFIGS as SPARSE_RESNETS
from .sigopt_experiments import CONFIGS as SIGOPT_EXPERIMENTS
CONFIGS = dict()
CONFIGS.update(DEFAULT_BASE)
CONFIGS.update(SPARSE_RESNETS)
CONFIGS.update(SIGOPT_EXPERIMENTS)