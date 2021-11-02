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


class ReportMaxAccuracy:
    """
    Reports the maximum accuracy and its corresponding epoch during training.
    """
    def setup_experiment(self, config):
        super().setup_experiment(config)
        self.max_accuracy = 0
        self.max_accuracy_epoch = 0

    def validate(self):
        ret = super().validate()
        current_accuracy = ret["mean_accuracy"]
        if current_accuracy > self.max_accuracy:
            self.max_accuracy = current_accuracy
            self.max_accuracy_epoch = self.current_epoch
        ret.update(max_accuracy=self.max_accuracy)
        ret.update(max_accuracy_epoch=self.max_accuracy_epoch)
        return ret

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["setup_experiment"].append("ReportMaxAccuracy: initialization")
        eo["validate"].append("ReportMaxAccuracy: store running max accuracy")
        return eo
