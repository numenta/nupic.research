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
import tensorflow as tf
from tensorflow.python import keras


class SparseWeights(keras.constraints.Constraint):
    """
    Sparse weights constraint.

    Constrains the weights to a fixed sparsity rate where a fixed number of
    weights are always zeros.

    :param percent_on:
      Percentage of weights that are allowed to be non-zero. Default 0.5
    :type sparsity: float
    """

    def __init__(self, percent_on=0.5, name=None):
        assert 0.0 < percent_on < 1.0
        self.percent_on = percent_on
        self.name = name or "sparse_mask"
        self._built = False

    def _build(self, input_shape, dtype=tf.float32):
        """
        Called on the first iteration once the input shape is known
        :param input_shape: Input shape including batch size
        """
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            non_zeros = int(round(input_shape[-1].value * self.percent_on))

            # Create random mask with k elements set to 1, all other elements set to 0
            values = tf.random_uniform(input_shape)
            top_k, _ = tf.math.top_k(input=values, k=non_zeros, sorted=False)
            kth = tf.reduce_min(top_k, axis=1, keepdims=True)
            mask = tf.cast(tf.greater_equal(values, kth), dtype=dtype)
            self.mask = tf.get_variable(
                self.name,
                initializer=mask,
                trainable=False,
                synchronization=tf.VariableSynchronization.NONE,
            )
            keras.backend.track_variable(self.mask)
            self._built = True

    def __call__(self, w):
        if not self._built:
            self._build(w.shape, dtype=w.dtype)

        return w * self.mask

    def get_config(self):
        return {"percent_on": self.percent_on, "name": self.name}
