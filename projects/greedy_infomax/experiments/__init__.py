from .default_base import CONFIGS as DEFAULT_BASE
from .sparse_resnets import CONFIGS as SPARSE_RESNETS
CONFIGS = dict()
CONFIGS.update(DEFAULT_BASE)
CONFIGS.update(SPARSE_RESNETS)