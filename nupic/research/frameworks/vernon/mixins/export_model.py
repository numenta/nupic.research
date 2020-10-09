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

import copy


class ExportModel:
    """
    Allows importing checkpoints from one type of model into another by first
    importing it into the original model then exporting to the new model.
    """
    @classmethod
    def create_model(cls, config, device):
        prev_config = copy.copy(config["prev_config"])
        if "checkpoint_file" in config:
            prev_config["checkpoint_file"] = config["checkpoint_file"]

        prev_model = super().create_model(prev_config, device)

        model_class = config["model_class"]
        model_args = config["model_args"]
        model = model_class(**model_args).to(device)

        export = config["export_model_fn"]
        export(prev_model, model)

        return model

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["create_model"].insert(0, "ExportModel: Set to create previous model")
        eo["create_model"].append("ExportModel: Export previous modell")
        return eo
