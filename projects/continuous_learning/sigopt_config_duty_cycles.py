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

sigopt_config = dict(
    name="GSC_duty_cycle_freezing",
    project="continuous_learning",
    observation_budget=200,
    parallel_bandwidth=1,
    parameters=[
        dict(
            name="cnn1_size",
            type="int",
            bounds=dict(min=64, max=512)
        ),
        dict(
            name="cnn2_size",
            type="int",
            bounds=dict(min=64, max=512)
        ),
        dict(
            name="cnn1_percent_on",
            type="double",
            bounds=dict(min=0.02, max=0.5)
        ),
        dict(
            name="cnn1_wt_sparsity",
            type="double",
            bounds=dict(min=0.05, max=0.8)
        ),
        dict(
            name="cnn2_percent_on",
            type="double",
            bounds=dict(min=0.02, max=0.5)
        ),
        dict(
            name="cnn2_wt_sparsity",
            type="double",
            bounds=dict(min=0.05, max=0.8)
        ),
        dict(
            name="linear1_n",
            type="int",
            bounds=dict(min=500, max=3000)
        ),
        dict(
            name="linear1_percent_on",
            type="double",
            bounds=dict(min=0.03, max=0.5)
        ),
        dict(
            name="linear1_weight_sparsity",
            type="double",
            bounds=dict(min=0.01, max=0.5)
        ),
        dict(
            name="linear2_percent_on",
            type="double",
            bounds=dict(min=0.03, max=0.5)
        ),
        dict(
            name="linear2_weight_sparsity",
            type="double",
            bounds=dict(min=0.01, max=0.5)
        ),
        dict(
            name="duty_cycle_period",
            type="int",
            bounds=dict(min=100, max=15000)
        ),
        dict(
            name="freeze_output",
            type="categorical",
            categorical_values=["True", "False"]
        ),
        dict(
            name="freeze_pct",
            type="int",
            bounds=dict(min=5, max=95),
        )
    ],
    metrics=[
        dict(
            name="area_under_curve",
            objective="maximize"
        )
    ]
)
