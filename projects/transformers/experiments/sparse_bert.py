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
Base Transformers Experiment configuration.
"""

from copy import deepcopy

from callbacks import SparsifyFCLayersCallback

from .base import debug_bert

sparse_debug_bert = deepcopy(debug_bert)
sparse_debug_bert.update(

    run_name="sparsity=0.2_debug_bert",
    # Model Arguments
    trainer_callbacks=[SparsifyFCLayersCallback(sparsity=0.80)]

)

# Export configurations in this file
CONFIGS = dict(
    sparse_debug_bert=sparse_debug_bert,
)
