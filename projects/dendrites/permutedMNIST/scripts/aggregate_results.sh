#!/bin/bash

RESULTS_DIR=~/nta/dendrites_results/mlp

python aggregate_ray_tune.py -d ${RESULTS_DIR}/three_layer_mlp_10_onehot
python aggregate_ray_tune.py -d ${RESULTS_DIR}/three_layer_mlp_10_centroid

# python aggregate_ray_tune.py -d ${RESULTS_DIR}/three_layer_mlp_10_onehot_sparse
python aggregate_ray_tune.py -d ${RESULTS_DIR}/three_layer_zero_segment_10_centroid_sparse_

# python aggregate_ray_tune.py -d ${RESULTS_DIR}/three_layer_mlp_10_onehot_dense_kw
python aggregate_ray_tune.py -d ${RESULTS_DIR}/three_layer_zero_segment_10_centroid_dense_kw_

# python aggregate_ray_tune.py -d ${RESULTS_DIR}/three_layer_mlp_10_onehot_sparse_kw
python aggregate_ray_tune.py -d ${RESULTS_DIR}/three_layer_zero_segment_10_centroid_sparse_kw_

python aggregate_ray_tune.py -d ${RESULTS_DIR}/three_layer_zero_segment_30_centroid_dense_kw_
python aggregate_ray_tune.py -d ${RESULTS_DIR}/three_layer_zero_segment_50_centroid_dense_kw_
python aggregate_ray_tune.py -d ${RESULTS_DIR}/three_layer_zero_segment_100_centroid_dense_kw_



python combine_csv.py -A ${RESULTS_DIR}/three_layer_mlp_10_onehot/aggregate_results.csv \
                         ${RESULTS_DIR}/three_layer_mlp_10_centroid/aggregate_results.csv \
                         ${RESULTS_DIR}/three_layer_zero_segment_10_centroid_sparse_/aggregate_results.csv \
                         ${RESULTS_DIR}/three_layer_zero_segment_10_centroid_dense_kw_/aggregate_results.csv \
                         ${RESULTS_DIR}/three_layer_zero_segment_10_centroid_sparse_kw_/aggregate_results.csv \
                      -O ${RESULTS_DIR}/all_results.csv

python combine_csv.py -A ${RESULTS_DIR}/three_layer_zero_segment_10_centroid_dense_kw_/aggregate_results.csv \
                         ${RESULTS_DIR}/three_layer_zero_segment_30_centroid_dense_kw_/aggregate_results.csv \
                         ${RESULTS_DIR}/three_layer_zero_segment_50_centroid_dense_kw_/aggregate_results.csv \
                         ${RESULTS_DIR}/three_layer_zero_segment_100_centroid_dense_kw_/aggregate_results.csv \
                         -O ${RESULTS_DIR}/n_task_scan_results.csv