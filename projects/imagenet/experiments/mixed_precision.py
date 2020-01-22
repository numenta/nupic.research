#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see http://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#
import copy
from .default import DEFAULT100, DEFAULT1000

MIXED_PRECISION_1000 = copy.deepcopy(DEFAULT1000)
MIXED_PRECISION_1000.update(
    batch_size=288,
    val_batch_size=288,
    mixed_precision=True,
    mixed_precision_args=dict(
        # See https://nvidia.github.io/apex/amp.html#opt-levels
        opt_level="O1",
        loss_scale=128.0,
    )
)

MIXED_PRECISION_100 = copy.deepcopy(DEFAULT100)
MIXED_PRECISION_100.update(
    batch_size=288,
    val_batch_size=288,
    mixed_precision=True,
    mixed_precision_args=dict(
        # See https://nvidia.github.io/apex/amp.html#opt-levels
        opt_level="O1",
        loss_scale=128.0,
    )
)

# Export all configurations
CONFIGS = dict(
    mixed_precision_100=MIXED_PRECISION_100,
    mixed_precision_1000=MIXED_PRECISION_1000,
)
