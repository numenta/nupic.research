# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
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
import abc

import numpy as np
import tensorflow as tf
from tensorflow import keras


def compute_kwinners(x, k, duty_cycles, boost_strength):
    r"""
    Use the boost strength to compute a boost factor for each unit represented
    in x. These factors are used to increase the impact of each unit to improve
    their chances of being chosen. This encourages participation of more columns
    in the learning process.

    The boosting function is a curve defined as: boost_factors = exp[ -
    boost_strength * (dutyCycle - target_density)] Intuitively this means that
    units that have been active (i.e. in the top-k) at the target activation
    level have a boost factor of 1, meaning their activity is not boosted.
    Columns whose duty cycle drops too much below that of their neighbors are
    boosted depending on how infrequently they have been active. Unit that has
    been active more than the target activation level have a boost factor below
    1, meaning their activity is suppressed and they are less likely to be in
    the top-k.

    Note that we do not transmit the boosted values. We only use boosting to
    determine the winning units.

    The target activation density for each unit is k / number of units. The
    boostFactor depends on the dutyCycle via an exponential function:

          boostFactor
              ^
              |
              |\
              | \
        1  _  |  \
              |    _
              |      _ _
              |          _ _ _ _
              +--------------------> dutyCycle
                 |
            target_density

    :param x:
        Current activity of each unit.

    :param k:
        The activity of the top k units will be allowed to remain, the rest are
        set to zero.

    :param duty_cycles:
        The averaged duty cycle of each unit.

    :param boost_strength:
        A boost strength of 0.0 has no effect on x.

    :return:
        A tensor representing the activity of x after k-winner take all.
    """
    k = tf.convert_to_tensor(k, dtype=tf.int32)
    boost_strength = tf.math.maximum(boost_strength, 0.0)
    batch_size = x.shape[0]
    n = tf.reduce_prod(x.shape[1:])
    target_density = tf.cast(k / n, tf.float32)
    boost_factors = tf.exp((target_density - duty_cycles) * boost_strength)
    boosted = x * boost_factors

    # Take the boosted version of the input x, find the top k winners.
    # Compute an output that contains the values of x corresponding to the top k
    # boosted values
    boosted = tf.reshape(boosted, [batch_size, -1])
    flat_x = tf.reshape(x, [batch_size, -1])
    top_k, indices = tf.math.top_k(input=boosted, k=k, sorted=False)
    range = tf.expand_dims(tf.range(0, tf.shape(indices)[0]), 1)
    range = tf.tile(range, [1, k])
    full_indices = tf.concat([tf.expand_dims(range, -1),
                              tf.expand_dims(indices, -1)], axis=2)
    full_indices = tf.reshape(full_indices, [-1, 2])

    updates = tf.gather_nd(params=flat_x, indices=full_indices)
    res = tf.scatter_nd(indices=full_indices, updates=updates, shape=flat_x.shape)
    return tf.reshape(res, x.shape)


class KWinnersBase(keras.layers.Layer, metaclass=abc.ABCMeta):
    """
    Base KWinners class.

    :param percent_on:
        The activity of the top k = percent_on * number of input units will be
        allowed to remain, the rest are set to zero.
    :type percent_on: float

    :param k_inference_factor:
        During inference (training=False) we increase percent_on by this factor.
        percent_on * k_inference_factor must be strictly less than 1.0, ideally
        much lower than 1.0
    :type k_inference_factor: float

    :param boost_strength:
        boost strength (0.0 implies no boosting). Must be >= 0.0
    :type boost_strength: float

    :param boost_strength_factor:
        Boost strength factor to use [0..1]
    :type boost_strength_factor: float

    :param duty_cycle_period:
        The period used to calculate duty cycles
    :type duty_cycle_period: int

    :param kwargs:
        Additional args passed to :class:`keras.layers.Layer`
    :type kwargs: dict
    """

    def __init__(
        self,
        percent_on,
        k_inference_factor,
        boost_strength,
        boost_strength_factor,
        duty_cycle_period,
        **kwargs
    ):
        super(KWinnersBase, self).__init__(**kwargs)
        self.percent_on = percent_on
        self.percent_on_inference = percent_on * k_inference_factor
        self.k_inference_factor = k_inference_factor
        self.learning_iterations = 0
        self.n = 0
        self.k = 0
        self.k_inference = 0

        # Boosting related parameters
        self.boost_strength = boost_strength
        self.boost_strength_factor = boost_strength_factor
        self.duty_cycle_period = duty_cycle_period
        self.duty_cycles = None

    def get_config(self):
        config = {
            "percent_on": self.percent_on,
            "k_inference_factor": self.k_inference_factor,
            "boost_strength": self.boost_strength,
            "boost_strength_factor": self.boost_strength_factor,
            "duty_cycle_period": self.duty_cycle_period,
        }
        config.update(super(KWinnersBase, self).get_config())
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

    @abc.abstractmethod
    def update_duty_cycle(self, x):
        r"""
        Updates our duty cycle estimates with the new value. Duty cycles are
        updated according to the following formula:

        .. math::
            dutyCycle = \frac{dutyCycle \times \left( period - batchSize \right)
                                + newValue}{period}

        :param x:
            Current activity of each unit
        """
        raise NotImplementedError

    def update_boost_strength(self):
        """
        Update boost strength using given strength factor during training.
        """
        self.boost_strength = keras.backend.in_train_phase(
            self.boost_strength * self.boost_strength_factor, self.boost_strength
        )


class KWinners2d(KWinnersBase):
    """
    Applies K-Winner function to Conv2D input tensor.

    :param percent_on:
        The activity of the top k = percent_on * number of input units will be
        allowed to remain, the rest are set to zero.
    :type percent_on: float

    :param k_inference_factor:
        During inference (training=False) we increase percent_on by this factor.
        percent_on * k_inference_factor must be strictly less than 1.0, ideally much
        lower than 1.0
    :type k_inference_factor: float

    :param boost_strength:
        boost strength (0.0 implies no boosting).
    :type boost_strength: float

    :param boost_strength_factor:
        Boost strength factor to use [0..1]
    :type boost_strength_factor: float

    :param duty_cycle_period:
        The period used to calculate duty cycles
    :type duty_cycle_period: int

    :param data_format:
        one of `channels_first` (default) or `channels_last`. The ordering of
        the dimensions in the inputs. `channels_last` corresponds to inputs with
        shape `(batch, height, width, channels)` while `channels_first` corresponds
        to inputs with shape `(batch, channels, height, width)`.
        Similar to `data_format` argument in :class:`keras.layers.Conv2D`.
    :type data_format: str

    :param kwargs:
        Additional args passed to :class:`keras.layers.Layer`
    :type kwargs: dict
    """

    def __init__(
        self,
        percent_on=0.1,
        k_inference_factor=1.5,
        boost_strength=1.0,
        boost_strength_factor=0.9,
        duty_cycle_period=1000,
        data_format="channel_first",
        **kwargs
    ):
        super(KWinners2d, self).__init__(
            percent_on=percent_on,
            k_inference_factor=k_inference_factor,
            boost_strength=boost_strength,
            boost_strength_factor=boost_strength_factor,
            duty_cycle_period=duty_cycle_period,
            **kwargs
        )
        self.data_format = data_format
        if self.data_format == "channel_first":
            # (batch, channels, height, width)
            self.channel_axis = 1
        else:
            # (batch, height, width, channels)
            self.channel_axis = 3

        # Not know until `build`
        self.channels = None
        self.scale_factor = None

    def build(self, input_shape):
        super(KWinners2d, self).build(input_shape=input_shape)
        shape = input_shape.as_list()
        self.channels = shape[self.channel_axis]

        duty_cycles_shape = [1] * 4
        duty_cycles_shape[self.channel_axis] = self.channels
        self.duty_cycles = self.add_variable(
            name="duty_cycles",
            shape=duty_cycles_shape,
            initializer=tf.zeros_initializer,
            trainable=False,
        )
        self.n = np.prod(shape[1:])
        self.k = round(self.n * self.percent_on)
        self.k_inference = round(self.k * self.k_inference_factor)

        del shape[self.channel_axis]  # Remove channel dim
        del shape[0]  # Remove batch dim
        self.scale_factor = np.prod(shape, dtype=np.float32)

    def get_config(self):
        config = {
            "data_format": self.data_format
        }
        config.update(super(KWinners2d, self).get_config())
        return config

    def update_duty_cycle(self, x):
        shape = x.shape.as_list()
        batch_size = shape[0]
        self.learning_iterations += batch_size

        period = tf.minimum(self.duty_cycle_period, self.learning_iterations)
        period = tf.cast(period, tf.float32)

        # Scale all dims but the channel dim
        axis = [0, 1, 2, 3]
        del axis[self.channel_axis]
        count = tf.reduce_sum(tf.cast(x > 0, tf.float32), axis=axis) / self.scale_factor

        duty_cycles = self.duty_cycles * (period - batch_size)
        duty_cycles = tf.reshape(duty_cycles, [-1]) + count
        return duty_cycles / period

    def call(self, inputs, training=None, **kwargs):
        k = keras.backend.in_test_phase(self.k_inference, self.k,
                                        training=training)

        x = compute_kwinners(x=inputs, k=k, duty_cycles=self.duty_cycles,
                             boost_strength=self.boost_strength)

        self.duty_cycles = keras.backend.in_train_phase(
            self.update_duty_cycle(x), self.duty_cycles, training=training
        )
        return x


class KWinners(KWinnersBase):
    """Applies K-Winner function to the input tensor.

    :param percent_on:
      The activity of the top k = percent_on * n will be allowed to remain, the
      rest are set to zero.
    :type percent_on: float

    :param k_inference_factor:
      During inference (training=False) we increase percent_on by this factor.
      percent_on * k_inference_factor must be strictly less than 1.0, ideally much
      lower than 1.0
    :type k_inference_factor: float

    :param boost_strength:
      boost strength (0.0 implies no boosting).
    :type boost_strength: float

    :param boost_strength_factor:
      Boost strength factor to use [0..1]
    :type boost_strength_factor: float

    :param duty_cycle_period:
      The period used to calculate duty cycles
    :type duty_cycle_period: int

    :param kwargs:
      Additional args passed to :class:`keras.layers.Layer`
    :type kwargs: dict
    """

    def __init__(
        self,
        percent_on,
        k_inference_factor=1.0,
        boost_strength=1.0,
        boost_strength_factor=1.0,
        duty_cycle_period=1000,
        **kwargs,
    ):
        super(KWinners, self).__init__(
            percent_on=percent_on,
            k_inference_factor=k_inference_factor,
            boost_strength=boost_strength,
            boost_strength_factor=boost_strength_factor,
            duty_cycle_period=duty_cycle_period,
            **kwargs)

    def build(self, input_shape):
        super(KWinners, self).build(input_shape=input_shape)
        self.n = input_shape[-1].value
        self.k = round(self.n * self.percent_on)
        self.k_inference = round(self.k * self.k_inference_factor)

        self.duty_cycles = self.add_variable(
            name="duty_cycles",
            shape=[self.n],
            initializer=tf.zeros_initializer,
            trainable=False,
        )

    def update_duty_cycle(self, inputs):
        r"""
        .. math::
            dutyCycle = \frac{dutyCycle \times \left( period - batchSize \right)
                        + newValue}{period}

        """
        batch_size = inputs.shape[0].value
        self.learning_iterations += batch_size
        period = tf.minimum(self.duty_cycle_period, self.learning_iterations)
        period = tf.cast(period, tf.float32)
        count = tf.reduce_sum(tf.cast(inputs > 0, tf.float32), axis=0)
        return ((self.duty_cycles * (period - batch_size)) + count) / period

    def call(self, inputs, training=None, **kwargs):
        k = keras.backend.in_test_phase(x=self.k_inference,
                                        alt=self.k,
                                        training=training)
        kwinners = compute_kwinners(
            x=inputs, k=k, duty_cycles=self.duty_cycles,
            boost_strength=self.boost_strength)
        self.duty_cycles = keras.backend.in_train_phase(
            x=self.update_duty_cycle(kwinners),
            alt=self.duty_cycles,
            training=training
        )
        return kwinners
