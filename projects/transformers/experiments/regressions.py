# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

from copy import deepcopy

from .trifecta import tiny_bert_trifecta_300k

"""
Regression tests configs: As the Transformers project is updated the results from
these configs should be reproducible.
"""

# run it on p3.2xlarge - results available on nupic-research in wandb
tiny_bert_trifecta_50k = deepcopy(tiny_bert_trifecta_300k)
tiny_bert_trifecta_50k.update(
    max_steps=50000,
)


CONFIGS = dict(
    tiny_bert_trifecta_50k=tiny_bert_trifecta_50k,
)
