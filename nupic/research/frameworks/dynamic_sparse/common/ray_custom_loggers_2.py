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

import codecs
import csv
import logging
import os
import pickle
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ray.tune.logger import CSVLogger, JsonLogger, Logger
from ray.tune.result import TIME_TOTAL_S, TIMESTEPS_TOTAL, TRAINING_ITERATION
from ray.tune.util import flatten_dict

logger = logging.getLogger(__name__)


# Note:
# This file is introduced as a potential replacement for `ray_custom_loggers.py`.
# Therein, the code is written for tensorflow>=1.13.1; however, we've moved to using
# tensorflow 2.0 and up.
#


def record_tf_values(result, path, step, num_hist_bins=None):
    """
    Tensorboard will display under scalars, images, histograms, & distributions
    tabs.

    We manually generate the summary objects from raw data passed from tune
    via the logger.

    Currently supports:
        * Scalar (any prefix, already supported by TFLogger)
        * Seaborn plot ("seaborn_" prefix; see below for configuration details)
        * Image ("img_" prefix)
        * Histograms ("hist_" prefix)

    ```
    writer = tf.summary.create_file_writer("/some/path")
    result = {
        "my_scalor": 4,
        "hist_some_array": [1, 2, 3, 4],
        "seaborn_dict": dict(
            plot_type="lineplot",
            x=[1, 2, 3],
            y=[1, 2, 3],
        ),
        "img_some_array": np.random.rand(3, 3, 3, 3)
    }

    with writer.as_default():
        record_tf_values(result=result, path=["ray", "tune"], step=1)
    write.flush()
    """
    for attr, value in result.items():
        if value is not None:
            if attr.startswith("seaborn_"):

                # Value should be a dict which defines the plot to make. For example:
                #
                #   value = {
                #
                #      # Plot setup.
                #      plot_type: string,         # name of seaborn plotting function
                #      config: {...},             # (optional) passed to seaborn.set
                #      edit_axes_func: <callable> # (optional) edits axes e.g. set xlim
                #
                #      # Params -  to be passed to seaborn plotting method.
                #      data: pandas.DataFrame(...),
                #      x: <string>, # label of desired column of data
                #      y: <string>, # label of desired column of data , same size as x
                #      hue: <None or array like>  # same size as x and y
                #
                #   }
                #
                if not isinstance(value, dict):
                    continue

                # Get seaborn plot type and config.
                config = value.pop("config", {})
                plot_type = value.pop("plot_type", None)
                edit_axes_func = value.pop("edit_axes_func", lambda x: x)

                if not hasattr(sns, plot_type):
                    continue

                # Plot seaborn plot.
                plot_type = getattr(sns, plot_type)
                sns.set(**config)
                ax = plot_type(**value)
                edit_axes_func(ax)

                # Convert to figure to numpy array of an equivalent.
                # Save the plot to a PNG in memory.
                stream = BytesIO()
                plt.savefig(stream, format="png")
                stream.seek(0)

                # Convert PNG buffer to TF image
                image_tf = tf.image.decode_png(stream.getvalue(), channels=4)

                # Add the batch dimension
                image_tf = tf.expand_dims(image_tf, 0)

                # Save array as an image.
                name = "/".join(path + [attr])
                tf.summary.image(name=name, data=image_tf, step=step)

            if attr.startswith("img_"):

                # Convert to numpy array and save.
                name = "/".join(path + [attr])
                value_np = np.array(value)
                tf.summary.image(name=name, data=value_np, step=step)

            elif attr.startswith("hist_"):

                # Convert to a numpy array
                name = "/".join(path + [attr])
                value_np = np.array(value)
                tf.summary.histogram(
                    name=name, data=value_np, step=step, buckets=num_hist_bins)

            else:
                if type(value) in [int, float, np.float32, np.float64, np.int32]:
                    tf.summary.scalar(
                        name="/".join(path + [attr]), data=value, step=step
                    )
                elif type(value) is dict:
                    record_tf_values(result=value, path=path + [attr], step=step)


class TFLoggerPlus(Logger):
    """Tensorboard logger that supports histograms and images based on key
    prefixes 'hist_' and 'img_'.

    Pass instead of TFLogger e.g. tune.run(..., loggers=(JsonLogger,
    CSVLogger, TFLoggerPlus))
    """

    def _init(self):
        try:
            global tf
            if "RLLIB_TEST_NO_TF_IMPORT" in os.environ:
                logger.warning("Not importing TensorFlow for test purposes")
                tf = None
            else:
                import tensorflow
                tf = tensorflow

        except ImportError:
            logger.warning(
                "Couldn't import TensorFlow - " "disabling TensorBoard logging."
            )
        self._file_writer = tf.summary.create_file_writer(self.logdir)

    def on_result(self, result):

        # Copy and remove extraneous results.
        result = result.copy()
        for k in ["config", "pid", "timestamp", TIME_TOTAL_S, TRAINING_ITERATION]:
            if k in result:
                del result[k]  # not useful to tf log these

        # Get and record training iteration (i.e. step).
        step = result[TRAINING_ITERATION] or result.get(TIMESTEPS_TOTAL)
        with self._file_writer.as_default():
            record_tf_values(
                result={"training_iteration": step},
                path=["ray", "tune"],
                step=step
            )
            self.flush()

        # Record results.
        with self._file_writer.as_default():
            record_tf_values(
                result=result,
                path=["ray", "tune"],
                step=step
            )
            self.flush()

    def flush(self):
        self._file_writer.flush()

    def close(self):
        self._file_writer.close()


class CSVLoggerPlus(CSVLogger):

    # Define object types in which to save in pickled form.
    pickle_types = (np.ndarray, pd.DataFrame)

    def on_result(self, result):
        tmp = result.copy()
        if "config" in tmp:
            del tmp["config"]
        result = flatten_dict(tmp, delimiter="/")
        if self._csv_out is None:
            self._csv_out = csv.DictWriter(self._file, result.keys())
            if not self._continuing:
                self._csv_out.writeheader()

        encode_results = {}
        for k, v in result.items():
            if k not in self._csv_out.fieldnames:
                continue

            if isinstance(v, self.pickle_types):
                v = pickle.dumps(v)
                v = codecs.encode(v, "base64").decode()
            encode_results[k] = v

        self._csv_out.writerow(encode_results)
        self._file.flush()


DEFAULT_LOGGERS = (JsonLogger, CSVLoggerPlus, TFLoggerPlus)


if __name__ == "__main__":

    # ---------------------
    # Test of TFLoggerPlus
    # ---------------------

    from tempfile import TemporaryDirectory

    tempdir = TemporaryDirectory()
    logger = TFLoggerPlus(config={}, logdir=tempdir.name)
    result = {
        "hello": 4,
        "hist_": [1, 2, 3, 4],
        "seaborn_test": dict(
            plot_type="lineplot",
            x=[1, 2, 3],
            y=[1, 2, 3],
        ),
        "img_test": np.random.rand(3, 3, 3, 3)
    }

    with logger._file_writer.as_default():
        record_tf_values(result=result, path=["ray", "tune"], step=1)

    print("Logged results successfully.")
    input("Try \ntensorboard --logdir={}".format(tempdir.name))
