#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
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
"""
This script is used to import GSC models pre-trained in pytorch into tensorflow
"""

import numpy as np
import tensorflow.keras.backend as K


def _reflatten_linear_weight(x):
    """
    In pytorch, the convolution layer outputs data in "channels first" format.
    This output is then flattened and fed into a linear layer.
    We need to convert the convolution output to "channel last" before it is
    flattened  and fed into the linear layer so they represent flattened
    "channel last" convolution output.

    :param x: Flattened conv weights trained using "channels first" format
    :type x: numpy.ndarray

    :return: Flattened conv weights using "channels last" format
    :rtype numpy.ndarray
    """
    output_size = x.shape[0]

    # restore original convolution shape based on the previous conv layer
    x = x.reshape(-1, 64, 5, 5)

    # swap channel axis
    x = x.transpose(0, 2, 3, 1)

    # re-flatten
    x = x.reshape(output_size, -1)

    # final transpose
    x = x.transpose()
    return x


# Maps GSC variable names to pytorch state_dict keys
# For each tensorflow variable name returns a tuple with pytorch state_dict key
# and an optional transformation function
_GSC_SPARSE_MAP = {
    # CNN 1: channel last
    "cnn1/kernel:0": ("cnn1.weight", lambda x: np.transpose(x, [2, 3, 1, 0])),
    "cnn1/bias:0": ("cnn1.bias", None),

    "cnn1_batchnorm/moving_mean:0": ("cnn1_batchnorm.running_mean", None),
    "cnn1_batchnorm/moving_variance:0": ("cnn1_batchnorm.running_var", None),
    # "cnn1_batchnorm.num_batches_tracked"

    # "cnn1_kwinner/learning_iterations:0"
    "cnn1_kwinner/boost_strength:0": ("cnn1_kwinner.boost_strength", None),
    # duty_cycle: channel last
    "cnn1_kwinner/duty_cycles:0": ("cnn1_kwinner.duty_cycle",
                                   lambda x: np.transpose(x, [0, 2, 3, 1])),

    # CNN 2: channel last
    "cnn2/kernel:0": ("cnn2.weight", lambda x: np.transpose(x, [2, 3, 1, 0])),
    "cnn2/bias:0": ("cnn2.bias", None),

    "cnn2_batchnorm/moving_mean:0": ("cnn2_batchnorm.running_mean", None),
    "cnn2_batchnorm/moving_variance:0": ("cnn2_batchnorm.running_var", None),
    # "cnn2_batchnorm.num_batches_tracked"

    # "cnn2_kwinner/learning_iterations:0"
    "cnn2_kwinner/boost_strength:0": ("cnn2_kwinner.boost_strength", None),
    # duty_cycle: channel last
    "cnn2_kwinner/duty_cycles:0": ("cnn2_kwinner.duty_cycle",
                                   lambda x: np.transpose(x, [0, 2, 3, 1])),

    # Linear: re-flatten assuming channel last
    "linear/kernel:0": ("linear.module.weight", _reflatten_linear_weight),
    "linear/bias:0": ("linear.module.bias", None),
    # 'linear.zero_weights'

    "linear_bn/moving_mean:0": ("linear_bn.running_mean", None),
    "linear_bn/moving_variance:0": ("linear_bn.running_var", None),
    # "linear_bn.num_batches_tracked"

    "linear_kwinner/boost_strength:0": ("linear_kwinner.boost_strength", None),
    "linear_kwinner/duty_cycles:0": ("linear_kwinner.duty_cycle", None),
    # "linear_kwinner/learning_iterations:0"

    # Output
    "output/kernel:0": ("output.weight", np.transpose),
    "output/bias:0": ("output.bias", None),
}


def load_pytorch_weights(model_tf, model_pt, weights_map=None):
    """
    Update tensorflow model weights using pre-trained pytorch model
    :param model_tf: Clean tensorflow model
    :type model_tf: :class:`nupic.tensorflow.models.GSCSparseCNN`
    :param model_pt: Pre-trained pytorch model
    :type model_pt: :class:`nupic.torch.models.GSCSparseCNN`
    :param weights_map: Dictionay mapping tensorflow variables to pytorch state
    :type weights_map: dict
    """
    if weights_map is None:
        weights_map = _GSC_SPARSE_MAP
    state_dict = model_pt.state_dict()
    batch_values = []
    for var in model_tf.variables:
        name = var.name
        if name in weights_map:
            tensor, transform = weights_map[name]
            value = state_dict[tensor].data.numpy()

            if transform is not None:
                value = transform(value)
            batch_values.append((var, value))

    K.batch_set_value(batch_values)
