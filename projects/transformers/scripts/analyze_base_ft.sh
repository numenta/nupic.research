#!/bin/zsh

# CSV files below will be updated sequentially, then a markdown table gets written out
# after results for the last model in a group (like base or small) are added

# Bert base and bert base trifecta
BASE_CSV=~/nta/nupic.research/projects/transformers/results/base_condensed_finetuning_results.csv
BASE_MD=~/nta/nupic.research/projects/transformers/results/base_condensed_finetuning_results.md

rm ${BASE_CSV}

## Base bert

# Bert 100k
python ../export_finetuning_results.py \
                ~/nta/finetuning/finetuning_bert_100k_glue_get_info \
                ~/nta/finetuning/bert_100k_hp_chase \
                ~/nta/finetuning/bert_100k_hp_chase_mnli \
                --model_name bert_100k \
                --csv ${BASE_CSV} \
                --pretrained_model Dense

# Trifecta 80
python ../export_finetuning_results.py \
                ~/nta/finetuning/finetuning_bert_sparse_trifecta_100k_glue_get_info_nb \
                ~/nta/finetuning/trifecta_80_hp_chase \
                ~/nta/finetuning/trifecta_80_hp_chase_mnli \
                --model_name trifecta_80_condensed \
                --csv ${BASE_CSV} \
                --pretrained_model Trifecta_80

# Trifecta 85
python ../export_finetuning_results.py \
                ~/nta/finetuning/finetuning_bert_sparse_85_trifecta_100k_glue_get_info \
                ~/nta/finetuning/trifecta_85_hp_chase \
                ~/nta/finetuning/trifecta_85_hp_chase_mnli \
                --model_name trifecta_85_condensed \
                --csv ${BASE_CSV} \
                --pretrained_model Trifecta_85

# Trifecta 90
python ../export_finetuning_results.py \
                ~/nta/finetuning/finetuning_bert_sparse_90_trifecta_100k_glue_get_info \
                ~/nta/finetuning/trifecta_90_hp_chase \
                ~/nta/finetuning/trifecta_90_hp_chase_mnli \
                ~/nta/finetuning/trifecta_90_hp_chase_follow_up \
                --model_name trifecta_90_condensed \
                --csv ${BASE_CSV} \
                --pretrained_model Trifecta_90 \
                --md ${BASE_MD}