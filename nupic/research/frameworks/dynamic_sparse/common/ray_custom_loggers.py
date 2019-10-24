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
import distutils.version
import logging
import os
import pickle
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL.Image
import seaborn as sns
from ray.tune.logger import CSVLogger, JsonLogger, Logger
from ray.tune.result import TIME_TOTAL_S, TIMESTEPS_TOTAL, TRAINING_ITERATION
from ray.tune.util import flatten_dict

logger = logging.getLogger(__name__)


def to_tf_values(result, path, histo_bins=1000):
    """Adapted from [1], generate a list of tf.Summary.Value() objects that
    Tensorboard will display under scalars, images, histograms, & distributions
    tabs.

    We manually generate the summary objects from raw data passed from tune via the logger.

    Currently supports:
        * Scalar (any prefix, already supported by TFLogger)
        * Image ("img_" prefix)
        * Histograms ("hist_" prefix)

    [1] https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514#file-tensorboard_logging-py-L41  # noqa
    """
    values = []
    for attr, value in result.items():
        if value is not None:
            if attr.startswith("scatter_"):

                # Value should be a dict which may be unpacked to sns.scatterplot e.g.
                #
                #   value = {
                #      data: Dataframe
                #      x: String - col of data
                #      y: String - col of data, same size as x
                #      hue: None or array like, same size as x and y
                #   }
                #
                if not isinstance(value, dict):
                    continue

                # Plot scatter plot.
                seaborn_config = value.pop("seaborn_config", {})
                sns.set(**seaborn_config)
                ax = sns.scatterplot(**value)

                # Save to BytesIO stream.
                stream = BytesIO()
                canvas = ax.figure.canvas
                canvas.draw()
                (w, h) = canvas.get_width_height()
                pilimage = PIL.Image.frombytes("RGB", (w, h), canvas.tostring_rgb())
                pilimage.save(stream, "PNG")

                # Create an Image object
                img_sum = tf.Summary.Image(
                    encoded_image_string=stream.getvalue(), height=h, width=w
                )

                # Create a Summary value
                values.append(
                    tf.Summary.Value(tag="/".join(path + [attr]), image=img_sum)
                )
                plt.clf()

            if attr.startswith("img_"):
                # for nr, img in enumerate(value):
                # Write the image to a string
                s = BytesIO()
                img = np.array(value)
                plt.imsave(s, img, format="png")

                # Create an Image object
                img_sum = tf.Summary.Image(
                    encoded_image_string=s.getvalue(),
                    height=img.shape[0],
                    width=img.shape[1],
                )
                # Create a Summary value
                values.append(
                    tf.Summary.Value(tag="/".join(path + [attr]), image=img_sum)
                )
            elif attr.startswith("hist_"):
                # Convert to a numpy array
                value_np = np.array(value)

                # Create histogram using numpy
                counts, bin_edges = np.histogram(value_np, bins=histo_bins)

                # Fill fields of histogram proto
                hist = tf.HistogramProto()
                hist.min = float(np.min(value_np))
                hist.max = float(np.max(value_np))
                hist.num = int(np.prod(value_np.shape))
                hist.sum = float(np.sum(value_np))
                hist.sum_squares = float(np.sum(value_np ** 2))
                bin_edges = bin_edges[1:]

                # Add bin edges and counts
                for edge in bin_edges:
                    hist.bucket_limit.append(edge)
                for c in counts:
                    hist.bucket.append(c)

                # Create and write Summary
                values.append(tf.Summary.Value(tag="/".join(path + [attr]), histo=hist))

            else:
                if use_tf150_api:
                    type_list = [int, float, np.float32, np.float64, np.int32]
                else:
                    type_list = [int, float]
                if type(value) in type_list:
                    values.append(
                        tf.Summary.Value(
                            tag="/".join(path + [attr]), simple_value=value
                        )
                    )
                elif type(value) is dict:
                    values.extend(to_tf_values(value, path + [attr]))
    return values


class TFLoggerPlus(Logger):
    """Tensorboard logger that supports histograms and images based on key
    prefixes 'hist_' and 'img_'.

    Pass instead of TFLogger e.g. tune.run(..., loggers=(JsonLogger,
    CSVLogger, TFLoggerPlus))
    """

    def _init(self):
        try:
            global tf, use_tf150_api
            if "RLLIB_TEST_NO_TF_IMPORT" in os.environ:
                logger.warning("Not importing TensorFlow for test purposes")
                tf = None
            else:
                import tensorflow

                tf = tensorflow
                use_tf150_api = distutils.version.LooseVersion(
                    tf.VERSION
                ) >= distutils.version.LooseVersion("1.5.0")
        except ImportError:
            logger.warning(
                "Couldn't import TensorFlow - " "disabling TensorBoard logging."
            )
        self._file_writer = tf.summary.FileWriter(self.logdir)

    def on_result(self, result):
        tmp = result.copy()
        for k in ["config", "pid", "timestamp", TIME_TOTAL_S, TRAINING_ITERATION]:
            if k in tmp:
                del tmp[k]  # not useful to tf log these
        values = to_tf_values(tmp, ["ray", "tune"])
        train_stats = tf.Summary(value=values)
        t = result.get(TIMESTEPS_TOTAL) or result[TRAINING_ITERATION]
        self._file_writer.add_summary(train_stats, t)
        iteration_value = to_tf_values(
            {"training_iteration": result[TRAINING_ITERATION]}, ["ray", "tune"]
        )
        iteration_stats = tf.Summary(value=iteration_value)
        self._file_writer.add_summary(iteration_stats, t)
        self._file_writer.flush()

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
