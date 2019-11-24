# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
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

from nupic.research.frameworks.pytorch.models import LeSparseNet


def gsc_lesparsenet(
        input_shape=(1, 32, 32),
        cnn_out_channels=(64, 64),
        cnn_activity_percent_on=(0.095, 0.125),
        cnn_weight_percent_on=(0.5, 0.2),
        linear_n=(1000,),
        linear_activity_percent_on=(0.1,),
        linear_weight_percent_on=(0.1,),
        num_classes=12,
        boost_strength=1.5,
        boost_strength_factor=0.9,
        duty_cycle_period=1000,
        k_inference_factor=1.0,
        use_batch_norm=True,
        use_softmax=False,
        **config):

    return LeSparseNet(
        input_shape=input_shape,
        cnn_out_channels=cnn_out_channels,
        cnn_activity_percent_on=cnn_activity_percent_on,
        cnn_weight_percent_on=cnn_weight_percent_on,
        linear_n=linear_n,
        linear_activity_percent_on=linear_activity_percent_on,
        linear_weight_percent_on=linear_weight_percent_on,
        num_classes=num_classes,
        boost_strength=boost_strength,
        boost_strength_factor=boost_strength_factor,
        duty_cycle_period=duty_cycle_period,
        k_inference_factor=k_inference_factor,
        use_batch_norm=use_batch_norm,
        use_softmax=use_softmax,
        **config)


def mnist_lesparsenet(
        input_shape=(1, 28, 28),
        cnn_out_channels=(32, 64),
        cnn_activity_percent_on=(0.1, 0.2),
        cnn_weight_percent_on=(0.6, 0.45),
        linear_n=(700,),
        linear_activity_percent_on=(0.2,),
        linear_weight_percent_on=(0.2,),
        num_classes=10,
        boost_strength=1.5,
        boost_strength_factor=0.85,
        duty_cycle_period=1000,
        k_inference_factor=1.0,
        use_batch_norm=True,
        use_softmax=False,
        **config):

    return LeSparseNet(
        input_shape=input_shape,
        cnn_out_channels=cnn_out_channels,
        cnn_activity_percent_on=cnn_activity_percent_on,
        cnn_weight_percent_on=cnn_weight_percent_on,
        linear_n=linear_n,
        linear_activity_percent_on=linear_activity_percent_on,
        linear_weight_percent_on=linear_weight_percent_on,
        num_classes=num_classes,
        boost_strength=boost_strength,
        boost_strength_factor=boost_strength_factor,
        duty_cycle_period=duty_cycle_period,
        k_inference_factor=k_inference_factor,
        use_batch_norm=use_batch_norm,
        use_softmax=use_softmax,
        **config)
