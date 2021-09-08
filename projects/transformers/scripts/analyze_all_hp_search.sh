#!/bin/zsh

# Where best hyperparameters will be stored
HP_CONFIG_DIR=~/nta/nupic.research/projects/transformers/experiments/hp_finetuning

# Where current hyperparameter search data is stored. You may need to modify this 
# to point to your local hyperparameter search data directory.
HP_RESULTS_DIR=~/nta/hp_search
PYTHON_SCRIPT_DIR=~/nta/nupic.research/projects/transformers

# BERT_100K big and small tasks
python ${PYTHON_SCRIPT_DIR}/export_finetuning_hp_search_results.py \
            -d ${HP_RESULTS_DIR}/hp_search_finetuning_bert_100k_big_tasks \
            -c ${HP_CONFIG_DIR}/bert_100k \
            -n 0
python ${PYTHON_SCRIPT_DIR}/export_finetuning_hp_search_results.py \
            -d ${HP_RESULTS_DIR}/hp_search_finetuning_bert_100k_small_tasks \
            -c ${HP_CONFIG_DIR}/bert_100k \
            -n 0

# TRIFECTA 80
python ${PYTHON_SCRIPT_DIR}/export_finetuning_hp_search_results.py \
            -d ${HP_RESULTS_DIR}/hp_search_finetuning_trifecta_80_100k_big_tasks \
            -c ${HP_CONFIG_DIR}/trifecta_80 \
            -n 0
python ${PYTHON_SCRIPT_DIR}/export_finetuning_hp_search_results.py \
            -d ${HP_RESULTS_DIR}/hp_search_finetuning_trifecta_80_100k_small_tasks \
            -c ${HP_CONFIG_DIR}/trifecta_80 \
            -n 0

# TRIFECTA 85
python ${PYTHON_SCRIPT_DIR}/export_finetuning_hp_search_results.py \
            -d ${HP_RESULTS_DIR}/hp_search_finetuning_trifecta_85_100k_big_tasks \
            -c ${HP_CONFIG_DIR}/trifecta_85 \
            -n 0
python ${PYTHON_SCRIPT_DIR}/export_finetuning_hp_search_results.py \
            -d ${HP_RESULTS_DIR}/hp_search_finetuning_trifecta_85_100k_small_tasks \
            -c ${HP_CONFIG_DIR}/trifecta_85 \
            -n 0

# TRIFECTA 90
python ${PYTHON_SCRIPT_DIR}/export_finetuning_hp_search_results.py \
            -d ${HP_RESULTS_DIR}/hp_search_finetuning_trifecta_90_100k_big_tasks \
            -c ${HP_CONFIG_DIR}/trifecta_90 \
            -n 0
python ${PYTHON_SCRIPT_DIR}/export_finetuning_hp_search_results.py \
            -d ${HP_RESULTS_DIR}/hp_search_finetuning_trifecta_90_100k_small_tasks \
            -c ${HP_CONFIG_DIR}/trifecta_90 \
            -n 0

# Note, I did not hyperparameter search the 2x model because it was going to be expensive. Instead,
# I guessed what the hyperparams should be and finetuned once.

### BERTITIOS

PROXY_DICT='{"qqp": ["mnli", "qnli", "sst2"]}'

# BERT_SMALL_100K
python ${PYTHON_SCRIPT_DIR}/export_finetuning_hp_search_results.py \
            -d ${HP_RESULTS_DIR}/hp_search_finetuning_small_bert_100k_small_tasks \
            -c ${HP_CONFIG_DIR}/small_100k \
            -n 0
python ${PYTHON_SCRIPT_DIR}/export_finetuning_hp_search_results.py \
            -d ${HP_RESULTS_DIR}/hp_search_finetuning_small_bert_100k_qqp \
            -c ${HP_CONFIG_DIR}/small_100k \
            -n 0 \
            -p ${PROXY_DICT}


# BERT_SMALL_TRIFECTA_80
python ${PYTHON_SCRIPT_DIR}/export_finetuning_hp_search_results.py \
            -d ${HP_RESULTS_DIR}/hp_search_finetuning_small_bert_trifecta_100k_small_tasks \
            -c ${HP_CONFIG_DIR}/trifecta_small_80 \
            -n 0
python ${PYTHON_SCRIPT_DIR}/export_finetuning_hp_search_results.py \
            -d ${HP_RESULTS_DIR}/hp_search_finetuning_small_bert_trifecta_100k_qqp \
            -c ${HP_CONFIG_DIR}/trifecta_small_80 \
            -n 0 \
            -p ${PROXY_DICT}

# BERT_SMALL_TRIFECTA_85
python ${PYTHON_SCRIPT_DIR}/export_finetuning_hp_search_results.py \
            -d ${HP_RESULTS_DIR}/hp_search_finetuning_small_bert_trifecta_85_100k_small_tasks \
            -c ${HP_CONFIG_DIR}/trifecta_small_85 \
            -n 0
python ${PYTHON_SCRIPT_DIR}/export_finetuning_hp_search_results.py \
            -d ${HP_RESULTS_DIR}/hp_search_finetuning_small_bert_trifecta_85_100k_qqp \
            -c ${HP_CONFIG_DIR}/trifecta_small_85 \
            -n 0 \
            -p ${PROXY_DICT}

# BERT_SMALL_TRIFECTA_90
python ${PYTHON_SCRIPT_DIR}/export_finetuning_hp_search_results.py \
            -d ${HP_RESULTS_DIR}/hp_search_finetuning_small_bert_trifecta_90_100k_small_tasks \
            -c ${HP_CONFIG_DIR}/trifecta_small_90 \
            -n 0
python ${PYTHON_SCRIPT_DIR}/export_finetuning_hp_search_results.py \
            -d ${HP_RESULTS_DIR}/hp_search_finetuning_small_bert_trifecta_90_100k_qqp \
            -c ${HP_CONFIG_DIR}/trifecta_small_90 \
            -n 0 \
            -p ${PROXY_DICT}

# BERT_SMALL_TRIFECTA_2X
python ${PYTHON_SCRIPT_DIR}/export_finetuning_hp_search_results.py \
            -d ${HP_RESULTS_DIR}/hp_search_finetuning_small_bert_trifecta_2x_100k_small_tasks \
            -c ${HP_CONFIG_DIR}/trifecta_small_2x \
            -n 0
python ${PYTHON_SCRIPT_DIR}/export_finetuning_hp_search_results.py \
            -d ${HP_RESULTS_DIR}/hp_search_finetuning_small_bert_trifecta_2x_100k_qqp \
            -c ${HP_CONFIG_DIR}/trifecta_small_2x \
            -n 0 \
            -p ${PROXY_DICT}

# BERT_SMALL_TRIFECTA_4X
python ${PYTHON_SCRIPT_DIR}/export_finetuning_hp_search_results.py \
            -d ${HP_RESULTS_DIR}/hp_search_finetuning_small_bert_trifecta_4x_100k_small_tasks \
            -c ${HP_CONFIG_DIR}/trifecta_small_4x \
            -n 0
python ${PYTHON_SCRIPT_DIR}/export_finetuning_hp_search_results.py \
            -d ${HP_RESULTS_DIR}/hp_search_finetuning_small_bert_trifecta_4x_100k_qqp \
            -c ${HP_CONFIG_DIR}/trifecta_small_4x \
            -n 0 \
            -p ${PROXY_DICT}
