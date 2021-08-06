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


class MultiEvalSetsTrainerMixin:
    """
    Mixin to HF Trainer
    Gather eval metrics for multiple datasets, possibly from different
    distributions
    """

    def __init__(self, *args, **kwargs):
        """
        Add one single argument to 'trainer_mixin_args'

        :param eval_sets: List of names of datasets to evaluate on
        """

        super().__init__(*args, **kwargs)

        mixin_args = self.args.trainer_mixin_args

        self.eval_set_list = mixin_args.get("eval_sets")

    def evaluate(self, *args, **kwargs):

        output_metrics = []
        for eval_dataset in self.eval_dataset:
            output_metrics.append(super().evaluate(eval_dataset=eval_dataset))

        return output_metrics

