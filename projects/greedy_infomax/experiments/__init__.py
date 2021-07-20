from .default_base import CONFIGS as DEFAULT_BASE
from .sparse_resnets import CONFIGS as SPARSE_RESNETS
from .sigopt_experiments import CONFIGS as SIGOPT_EXPERIMENTS
from .small_sparse import CONFIGS as SMALL_SPARSE
CONFIGS = dict()
CONFIGS.update(DEFAULT_BASE)
CONFIGS.update(SPARSE_RESNETS)
CONFIGS.update(SIGOPT_EXPERIMENTS)
CONFIGS.update(SMALL_SPARSE)