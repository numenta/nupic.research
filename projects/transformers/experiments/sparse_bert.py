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
from .bert_replication import bert_100k

sparse_debug_bert = deepcopy(debug_bert)
sparse_debug_bert.update(

    # Model Arguments
    trainer_callbacks=[SparsifyFCLayersCallback(sparsity=0.80)]

)

sparse_bert_100k = deepcopy(bert_100k)
sparse_bert_100k.update(
    # run_name is optional, gets name from experiment name when not defined
    run_name="bert-steps_100k-sparsity_0.8",
    # Model Arguments
    trainer_callbacks=[SparsifyFCLayersCallback(sparsity=0.80)],
    overwrite_output_dir=False,

)


# Export configurations in this file
CONFIGS = dict(
    sparse_debug_bert=sparse_debug_bert,
    sparse_bert_100k=sparse_bert_100k
)
