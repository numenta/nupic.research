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


class ConstrainParameters(object):
    def setup_experiment(self, config):
        super().setup_experiment(config)
        self._constrain_parameters_modules = [
            module
            for module in self.model.modules()
            if hasattr(module, "constrain_parameters")
        ]

    def post_batch(self, *args, **kwargs):
        super().post_batch(*args, **kwargs)
        for module in self._constrain_parameters_modules:
            module.constrain_parameters()
