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
DENDRITES_DIR=~/nta/dendrites_results/dendrites

# Prototype context
python aggregate_ray_tune.py -d ${RESULTS_DIR}/three_layer_mlp_10_prototype
python aggregate_ray_tune.py -d ${RESULTS_DIR}/three_layer_zero_segment_10_prototype_sparse_
python aggregate_ray_tune.py -d ${RESULTS_DIR}/three_layer_zero_segment_10_prototype_dense_kw_
python aggregate_ray_tune.py -d ${RESULTS_DIR}/three_layer_zero_segment_10_prototype_sparse_kw_

# Onehot context
python aggregate_ray_tune.py -d ${RESULTS_DIR}/three_layer_mlp_10_onehot
python aggregate_ray_tune.py -d ${RESULTS_DIR}/three_layer_zero_segment_10_onehot_sparse_
python aggregate_ray_tune.py -d ${RESULTS_DIR}/three_layer_zero_segment_10_onehot_dense_kw_
python aggregate_ray_tune.py -d ${RESULTS_DIR}/three_layer_zero_segment_10_onehot_sparse_kw_

# Sparse binary context
python aggregate_ray_tune.py -d ${RESULTS_DIR}/three_layer_mlp_10_sparse_binary
python aggregate_ray_tune.py -d ${RESULTS_DIR}/three_layer_zero_segment_10_sparse_binary_sparse
python aggregate_ray_tune.py -d ${RESULTS_DIR}/three_layer_zero_segment_10_sparse_binary_dense_kw
python aggregate_ray_tune.py -d ${RESULTS_DIR}/three_layer_zero_segment_10_sparse_binary_sparse_kw

# Prototype context: scan n tasks
python aggregate_ray_tune.py -d ${RESULTS_DIR}/three_layer_zero_segment_10_prototype_dense_kw___
python aggregate_ray_tune.py -d ${RESULTS_DIR}/three_layer_zero_segment_30_prototype_dense_kw___
python aggregate_ray_tune.py -d ${RESULTS_DIR}/three_layer_zero_segment_50_prototype_dense_kw___
python aggregate_ray_tune.py -d ${RESULTS_DIR}/three_layer_zero_segment_100_prototype_dense_kw__

# Scan num epochs at each num_tasks for best prototype model
python aggregate_ray_tune.py -d ${RESULTS_DIR}/three_layer_zero_segment_30_prototype_dense_kw_scan_epoch
python aggregate_ray_tune.py -d ${RESULTS_DIR}/three_layer_zero_segment_50_prototype_dense_kw_scan_epoch
python aggregate_ray_tune.py -d ${RESULTS_DIR}/three_layer_zero_segment_100_prototype_dense_kw_scan_epoch

# Scan number of tasks using optimal n_epochs
python aggregate_ray_tune.py -d ${RESULTS_DIR}/three_layer_zero_segment_30_prototype_dense_kw_best_epoch
python aggregate_ray_tune.py -d ${RESULTS_DIR}/three_layer_zero_segment_50_prototype_dense_kw_best_epoch
python aggregate_ray_tune.py -d ${RESULTS_DIR}/three_layer_zero_segment_100_prototype_dense_kw_best_epoch

# Self-context: search n_epochs and learning_rate
python aggregate_ray_tune.py -d ${DENDRITES_DIR}/self_context_10

# Self-context: scan n_tasks
python aggregate_ray_tune.py -d ${DENDRITES_DIR}/self_context_10_
python aggregate_ray_tune.py -d ${DENDRITES_DIR}/self_context_25_
python aggregate_ray_tune.py -d ${DENDRITES_DIR}/self_context_50_
python aggregate_ray_tune.py -d ${DENDRITES_DIR}/self_context_100_


# Combine self-context runs scanning n_tasks
python combine_csv.py -A ${DENDRITES_DIR}/self_context_25_/aggregate_results.csv \
                         ${DENDRITES_DIR}/self_context_50_/aggregate_results.csv \
                         ${DENDRITES_DIR}/self_context_100_/aggregate_results.csv \
                         -O ${DENDRITES_DIR}/self_context_scan_n_tasks.csv


# Combine prototype fixing best num epochs at each n_tasks
python combine_csv.py -A ${RESULTS_DIR}/three_layer_zero_segment_30_prototype_dense_kw_best_epoch/aggregate_results.csv \
                         ${RESULTS_DIR}/three_layer_zero_segment_50_prototype_dense_kw_best_epoch/aggregate_results.csv \
                         ${RESULTS_DIR}/three_layer_zero_segment_100_prototype_dense_kw_best_epoch/aggregate_results.csv \
                         -O ${RESULTS_DIR}/prototype_scan_n_tasks_best_epochs.csv


# Combine prototype num epoch scans
python combine_csv.py -A ${RESULTS_DIR}/three_layer_zero_segment_30_prototype_dense_kw_scan_epoch/aggregate_results.csv \
                         ${RESULTS_DIR}/three_layer_zero_segment_50_prototype_dense_kw_scan_epoch/aggregate_results.csv \
                         ${RESULTS_DIR}/three_layer_zero_segment_100_prototype_dense_kw_scan_epoch/aggregate_results.csv \
                         -O ${RESULTS_DIR}/prototype_scan_n_epochs.csv


# Combine prototype n_tasks scan
python combine_csv.py -A ${RESULTS_DIR}/three_layer_zero_segment_10_prototype_dense_kw___/aggregate_results.csv \
                         ${RESULTS_DIR}/three_layer_zero_segment_30_prototype_dense_kw___/aggregate_results.csv \
                         ${RESULTS_DIR}/three_layer_zero_segment_50_prototype_dense_kw___/aggregate_results.csv \
                         ${RESULTS_DIR}/three_layer_zero_segment_100_prototype_dense_kw__/aggregate_results.csv \
                         -O ${RESULTS_DIR}/prototype_scan_n_tasks.csv


# Combine prototype 10 hyperparameter searches
python combine_csv.py -A ${RESULTS_DIR}/three_layer_mlp_10_prototype/aggregate_results.csv \
                         ${RESULTS_DIR}/three_layer_zero_segment_10_prototype_sparse_/aggregate_results.csv \
                         ${RESULTS_DIR}/three_layer_zero_segment_10_prototype_dense_kw_/aggregate_results.csv \
                         ${RESULTS_DIR}/three_layer_zero_segment_10_prototype_sparse_kw_/aggregate_results.csv \
                         -O ${RESULTS_DIR}/prototype_results.csv


# Combine onehot 10 hyperparameter searches
python combine_csv.py -A ${RESULTS_DIR}/three_layer_mlp_10_onehot/aggregate_results.csv \
                         ${RESULTS_DIR}/three_layer_zero_segment_10_onehot_sparse_/aggregate_results.csv \
                         ${RESULTS_DIR}/three_layer_zero_segment_10_onehot_dense_kw_/aggregate_results.csv \
                         ${RESULTS_DIR}/three_layer_zero_segment_10_onehot_sparse_kw_/aggregate_results.csv \
                         -O ${RESULTS_DIR}/onehot_results.csv


# Combine sparse binary hyperparameter searches
python combine_csv.py -A ${RESULTS_DIR}/three_layer_mlp_10_sparse_binary/aggregate_results.csv \
                         ${RESULTS_DIR}/three_layer_zero_segment_10_sparse_binary_sparse/aggregate_results.csv \
                         ${RESULTS_DIR}/three_layer_zero_segment_10_sparse_binary_dense_kw/aggregate_results.csv \
                         ${RESULTS_DIR}/three_layer_zero_segment_10_sparse_binary_sparse_kw/aggregate_results.csv \
                         -O ${RESULTS_DIR}/sparse_binary_results.csv
