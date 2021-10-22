#!/bin/bash

#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see htt"://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#


RESULTS_DIR=~/nta/dendrites_results/mlp

# Centroid context
python aggregate_ray_tune.py -d ${RESULTS_DIR}/three_layer_mlp_10_centroid
python aggregate_ray_tune.py -d ${RESULTS_DIR}/three_layer_zero_segment_10_centroid_sparse_
python aggregate_ray_tune.py -d ${RESULTS_DIR}/three_layer_zero_segment_10_centroid_dense_kw_
python aggregate_ray_tune.py -d ${RESULTS_DIR}/three_layer_zero_segment_10_centroid_sparse_kw_

# Onehot context
python aggregate_ray_tune.py -d ${RESULTS_DIR}/three_layer_mlp_10_onehot
python aggregate_ray_tune.py -d ${RESULTS_DIR}/three_layer_zero_segment_10_onehot_sparse_
python aggregate_ray_tune.py -d ${RESULTS_DIR}/three_layer_zero_segment_10_onehot_dense_kw_
python aggregate_ray_tune.py -d ${RESULTS_DIR}/three_layer_zero_segment_10_onehot_sparse_kw_

# Sparse binary context

# (these are running)

# Centroid context: scan n tasks
python aggregate_ray_tune.py -d ${RESULTS_DIR}/three_layer_zero_segment_30_centroid_dense_kw_
python aggregate_ray_tune.py -d ${RESULTS_DIR}/three_layer_zero_segment_50_centroid_dense_kw_
python aggregate_ray_tune.py -d ${RESULTS_DIR}/three_layer_zero_segment_100_centroid_dense_kw_

python aggregate_ray_tune.py -d ${RESULTS_DIR}/three_layer_zero_segment_30_centroid_dense_kw__
python aggregate_ray_tune.py -d ${RESULTS_DIR}/three_layer_zero_segment_30_centroid_dense_kw__
python aggregate_ray_tune.py -d ${RESULTS_DIR}/three_layer_zero_segment_100_centroid_dense_kw__


# python combine_csv.py -A ${RESULTS_DIR}/three_layer_mlp_10_onehot/aggregate_results.csv \
#                          ${RESULTS_DIR}/three_layer_mlp_10_centroid/aggregate_results.csv \
#                          ${RESULTS_DIR}/three_layer_zero_segment_10_centroid_sparse_/aggregate_results.csv \
#                          ${RESULTS_DIR}/three_layer_zero_segment_10_centroid_dense_kw_/aggregate_results.csv \
#                          ${RESULTS_DIR}/three_layer_zero_segment_10_centroid_sparse_kw_/aggregate_results.csv \
#                       -O ${RESULTS_DIR}/all_results.csv


python combine_csv.py -A ${RESULTS_DIR}/three_layer_zero_segment_10_centroid_dense_kw_/aggregate_results.csv \
                         ${RESULTS_DIR}/three_layer_zero_segment_30_centroid_dense_kw_/aggregate_results.csv \
                         ${RESULTS_DIR}/three_layer_zero_segment_50_centroid_dense_kw_/aggregate_results.csv \
                         ${RESULTS_DIR}/three_layer_zero_segment_100_centroid_dense_kw_/aggregate_results.csv \
                         ${RESULTS_DIR}/three_layer_zero_segment_30_centroid_dense_kw__/aggregate_results.csv \
                         ${RESULTS_DIR}/three_layer_zero_segment_50_centroid_dense_kw__/aggregate_results.csv \
                         ${RESULTS_DIR}/three_layer_zero_segment_100_centroid_dense_kw__/aggregate_results.csv \
                         -O ${RESULTS_DIR}/centroid_scan_n_tasks.csv


python combine_csv.py -A ${RESULTS_DIR}/three_layer_mlp_10_centroid/aggregate_results.csv \
                         ${RESULTS_DIR}/three_layer_zero_segment_10_centroid_sparse_/aggregate_results.csv \
                         ${RESULTS_DIR}/three_layer_zero_segment_10_centroid_dense_kw_/aggregate_results.csv \
                         ${RESULTS_DIR}/three_layer_zero_segment_10_centroid_sparse_kw_/aggregate_results.csv \
                         -O ${RESULTS_DIR}/centroid_results.csv

python combine_csv.py -A ${RESULTS_DIR}/three_layer_mlp_10_onehot/aggregate_results.csv \
                         ${RESULTS_DIR}/three_layer_mlp_10_onehot_sparse_/aggregate_results.csv \
                         ${RESULTS_DIR}/three_layer_mlp_10_onehot_dense_kw_/aggregate_results.csv \
                         ${RESULTS_DIR}/three_layer_mlp_10_onehot_sparse_kw_/aggregate_results.csv \
                         -O ${RESULTS_DIR}/onehot_results.csv