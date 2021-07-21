#!/bin/zsh

# Where best hyperparameters will be stored
HP_CONFIG_DIR=~/nta/nupic.research/projects/transformers/experiments/hp_finetuning

# Where current hyperparameter search data is stored. You may need to modify this 
# to point to your local hyperparameter search data directory.
HP_RESULTS_DIR=~/nta/hp_search


# BERT_100K big and small tasks
python ../export_finetuning_hp_search_results.py -d ${HP_RESULTS_DIR}/hp_search_finetuning_bert_100k_big_tasks -c ${HP_CONFIG_DIR}/bert_100k -n 0
python ../export_finetuning_hp_search_results.py -d ${HP_RESULTS_DIR}/hp_search_finetuning_bert_100k_small_tasks -c ${HP_CONFIG_DIR}/bert_100k -n 0

# TRIFECTA 80
python ../export_finetuning_hp_search_results.py -d ${HP_RESULTS_DIR}/hp_search_finetuning_trifecta_80_100k_big_tasks -c ${HP_CONFIG_DIR}/trifecta_80 -n 0
python ../export_finetuning_hp_search_results.py -d ${HP_RESULTS_DIR}/hp_search_finetuning_trifecta_80_100k_small_tasks -c ${HP_CONFIG_DIR}/trifecta_80 -n 0

# TRIFECTA 85
python ../export_finetuning_hp_search_results.py -d ${HP_RESULTS_DIR}/hp_search_finetuning_trifecta_85_100k_big_tasks -c ${HP_CONFIG_DIR}/trifecta_85 -n 0
python ../export_finetuning_hp_search_results.py -d ${HP_RESULTS_DIR}/hp_search_finetuning_trifecta_85_100k_small_tasks -c ${HP_CONFIG_DIR}/trifecta_85 -n 0

# TRIFECTA 90
python ../export_finetuning_hp_search_results.py -d ${HP_RESULTS_DIR}/hp_search_finetuning_trifecta_90_100k_big_tasks -c ${HP_CONFIG_DIR}/trifecta_90 -n 0
python ../export_finetuning_hp_search_results.py -d ${HP_RESULTS_DIR}/hp_search_finetuning_trifecta_90_100k_small_tasks -c ${HP_CONFIG_DIR}/trifecta_90 -n 0


################      TO DO 

# TRIFECTA_2X

# TRIFECTA_4X


### BERTITIOS

# BERT_SMALL_100K

# BERT_SMALL_TRIFECTA_80

# BERT_SMALL_TRIFECTA_85

# BERT_SMALL_TRIFECTA_90

# BERT_SMALL_TRIFECTA_2X

# BERT_SMALL_TRIFECTA_4X


