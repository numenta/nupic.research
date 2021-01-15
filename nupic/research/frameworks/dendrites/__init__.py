# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
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

from .dendrite_segments import DendriteSegments
from .dendritic_layers import (
    AbsoluteMaxGatingDendriticLayer,
    AbsoluteMaxGatingDendriticLayer2d,
    BiasingDendriticLayer,
    GatingDendriticLayer,
    GatingDendriticLayer2d,
)
from .plotting_utils import (
    plot_dendrite_activations,
    plot_dendrite_overlap_matrix,
    plot_entropy_distribution,
    plot_mean_selected_activations,
    plot_overlap_scores_distribution,
    plot_percent_active_dendrites,
)
from .utils import (
    dendrite_duty_cycle,
    dendrite_overlap,
    dendrite_overlap_matrix,
    entropy,
    mean_selected_activations,
    online_dendrite_duty_cycle,
    percent_active_dendrites,
)
