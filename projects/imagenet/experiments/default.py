#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
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
import copy

from .base import DEFAULT

"""
Imagenet Experiment configurations using the normal PyTorch learning schedule
"""

DEFAULT10 = copy.deepcopy(DEFAULT)
DEFAULT10.update(
    epochs=70,
    num_classes=10,
    model_args=dict(config=dict(num_classes=10, defaults_sparse=False)),

)

SPARSE10 = copy.deepcopy(DEFAULT10)
SPARSE10.update(
    model_args=dict(config=dict(num_classes=10, defaults_sparse=True)),
)

# Should get to 79.6% top-1 accuracy
DEFAULT100 = copy.deepcopy(DEFAULT)
DEFAULT100.update(
    epochs=70,
    num_classes=100,
    model_args=dict(config=dict(num_classes=100, defaults_sparse=False)),
)

# Should get to about 80.1%
SPARSE100 = copy.deepcopy(DEFAULT100)
SPARSE100.update(
    model_args=dict(config=dict(num_classes=100, defaults_sparse=True)),
)

# Should get to about 75%
DEFAULT1000 = copy.deepcopy(DEFAULT)
DEFAULT1000.update(
    epochs=90,
    num_classes=1000,
    model_args=dict(config=dict(num_classes=1000, defaults_sparse=False)),
)

# Should get to about 70.2%
SPARSE1000 = copy.deepcopy(DEFAULT100)
SPARSE1000.update(
    epochs=70,
    num_classes=1000,

    optimizer_args=dict(
        lr=0.5,
        weight_decay=1e-04,
        momentum=0.9,
        dampening=0,
        nesterov=True
    ),
    lr_scheduler_args=dict(
        gamma=0.25,
        step_size=15,
    ),

    model_args=dict(config=dict(num_classes=1000, defaults_sparse=True)),
)

# Export all configurations
CONFIGS = dict(
    default_10=DEFAULT10,
    default_sparse_10=SPARSE10,

    default_100=DEFAULT100,
    default_sparse_100=SPARSE100,

    default_1000=DEFAULT1000,
    default_sparse_1000=SPARSE1000,
)
