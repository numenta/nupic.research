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

import random

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import test_util
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model

from nupic.research.frameworks.tensorflow.layers.k_winners import KWinners, KWinners2d

tf.enable_eager_execution()

SEED = 18

# Tensorflow configuration.
# Make sure to use one thread in order to keep the results deterministic
CONFIG = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1,
    device_count={"CPU": 1},
)


class KWinnersTest(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        tf.set_random_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

    def setUp(self):
        self.x = tf.constant(
            [[1.0, 1.2, 1.1, 1.3, 1.0, 1.5, 1.0],
             [1.1, 1.0, 1.2, 1.0, 1.3, 1.0, 1.2]]
        )
        self.duty_cycle = tf.constant(1.0 / 3.0, shape=(2, 7))

    @test_util.run_all_in_graph_and_eager_modes
    def test_inference(self):
        """boost factor 0, k=3, batch size 2"""
        expected = np.zeros(self.x.shape)
        expected[0, 1] = 1.2
        expected[0, 3] = 1.3
        expected[0, 5] = 1.5
        expected[1, 2] = 1.2
        expected[1, 4] = 1.3
        expected[1, 6] = 1.2

        batch_size = self.x.shape[0]
        input_shape = self.x.shape[1:]
        n = np.prod(input_shape)

        k_winners = KWinners(percent_on=3 / n, boost_strength=0.0)
        inp = Input(batch_size=batch_size, shape=input_shape)
        out = k_winners(inp, training=False)
        model = Model(inp, out)

        with self.test_session(config=CONFIG):
            y = model.predict(self.x, steps=1, batch_size=batch_size)
            self.assertAllClose(y, expected)


class KWinners2dTest(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        tf.set_random_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

    def setUp(self):
        # Batch size 1
        x = np.ones((1, 3, 2, 2), dtype=np.float32)
        x[0, 0, 1, 0] = 1.1
        x[0, 0, 1, 1] = 1.2
        x[0, 1, 0, 1] = 1.2
        x[0, 2, 1, 0] = 1.3
        self.x = tf.constant(x)

        # Batch size 2
        x = np.ones((2, 3, 2, 2), dtype=np.float32)
        x[0, 0, 1, 0] = 1.1
        x[0, 0, 1, 1] = 1.2
        x[0, 1, 0, 1] = 1.2
        x[0, 2, 1, 0] = 1.3

        x[1, 0, 0, 0] = 1.4
        x[1, 1, 0, 0] = 1.5
        x[1, 1, 0, 1] = 1.6
        x[1, 2, 1, 1] = 1.7
        self.x2 = tf.constant(x)

    @test_util.run_all_in_graph_and_eager_modes
    def test_one(self):
        """Equal duty cycle, boost factor 0, k=4, batch size 1."""
        x = self.x
        expected = np.zeros_like(x)
        expected[0, 0, 1, 0] = 1.1
        expected[0, 0, 1, 1] = 1.2
        expected[0, 1, 0, 1] = 1.2
        expected[0, 2, 1, 0] = 1.3

        input_shape = x.shape[1:]
        n = np.prod(input_shape)
        k = 4

        with self.cached_session(config=CONFIG):
            k_winners = KWinners2d(percent_on=k / n, boost_strength=0.0)
            k_winners.build(x.shape)
            y = k_winners(x, training=True)
            self.assertAllClose(y, expected)

    @test_util.run_all_in_graph_and_eager_modes
    def test_two(self):
        """Equal duty cycle, boost factor 0, k=3."""
        x = self.x
        expected = np.zeros_like(x)
        expected[0, 0, 1, 1] = 1.2
        expected[0, 1, 0, 1] = 1.2
        expected[0, 2, 1, 0] = 1.3

        input_shape = x.shape[1:]
        n = np.prod(input_shape)
        k = 3

        with self.cached_session(config=CONFIG):
            k_winners = KWinners2d(percent_on=k / n, boost_strength=0.0)
            k_winners.build(x.shape)
            y = k_winners(x, training=True)
            self.assertAllClose(y, expected)

    @test_util.run_all_in_graph_and_eager_modes
    def test_three(self):
        """Equal duty cycle, boost factor=0, k=4, batch size=2."""
        x = self.x2
        expected = np.zeros_like(x)
        expected[0, 0, 1, 0] = 1.1
        expected[0, 0, 1, 1] = 1.2
        expected[0, 1, 0, 1] = 1.2
        expected[0, 2, 1, 0] = 1.3
        expected[1, 0, 0, 0] = 1.4
        expected[1, 1, 0, 0] = 1.5
        expected[1, 1, 0, 1] = 1.6
        expected[1, 2, 1, 1] = 1.7

        input_shape = x.shape[1:]
        n = np.prod(input_shape)
        k = 4

        with self.cached_session(config=CONFIG):
            k_winners = KWinners2d(percent_on=k / n, boost_strength=0.0)
            k_winners.build(x.shape)
            y = k_winners(x, training=True)
            self.assertAllClose(y, expected)

    @test_util.run_all_in_graph_and_eager_modes
    def test_four(self):
        """Equal duty cycle, boost factor=0, k=3, batch size=2."""
        x = self.x2
        expected = np.zeros_like(x)
        expected[0, 0, 1, 1] = 1.2
        expected[0, 1, 0, 1] = 1.2
        expected[0, 2, 1, 0] = 1.3
        expected[1, 1, 0, 0] = 1.5
        expected[1, 1, 0, 1] = 1.6
        expected[1, 2, 1, 1] = 1.7

        input_shape = x.shape[1:]
        n = np.prod(input_shape)
        k = 3

        with self.cached_session(config=CONFIG):
            k_winners = KWinners2d(percent_on=k / n, boost_strength=0.0)
            k_winners.build(x.shape)
            y = k_winners(x, training=True)
            self.assertAllClose(y, expected)

    @test_util.run_all_in_graph_and_eager_modes
    def test_five(self):
        x = self.x2
        expected = np.zeros_like(x)
        expected[0, 0, 1, 0] = 1.1
        expected[0, 0, 1, 1] = 1.2
        expected[0, 1, 0, 1] = 1.2
        expected[0, 2, 1, 0] = 1.3
        expected[1, 0, 0, 0] = 1.4
        expected[1, 1, 0, 0] = 1.5
        expected[1, 1, 0, 1] = 1.6
        expected[1, 2, 1, 1] = 1.7
        expected_dutycycles = tf.constant([1.5000, 1.5000, 1.0000]) / 4.0

        with self.cached_session(config=CONFIG):
            k_winners = KWinners2d(
                percent_on=0.333,
                boost_strength=1.0,
                k_inference_factor=0.5,
                boost_strength_factor=0.5,
                duty_cycle_period=1000,
            )
            k_winners.build(x.shape)
            y = k_winners(x, training=True)
            self.assertAllClose(y, expected)
            self.assertAllClose(k_winners.duty_cycles, expected_dutycycles)


if __name__ == "__main__":
    tf.test.main()
