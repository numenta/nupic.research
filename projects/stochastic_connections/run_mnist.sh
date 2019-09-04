#!/bin/bash

set -e -x

cd "$(dirname "$0")"
pwd

DATA_DIR=data/mnist_l0/$(python -c "import time; import uuid; print('{}-{}'.format(time.strftime('%Y%m%d-%H%M%S'), uuid.uuid1()))")
mkdir -p ./$DATA_DIR

python -u run_l0_experiment.py $DATA_DIR --epochs 30 # --redis-address localhost:52554
python -u plot_training.py $DATA_DIR
