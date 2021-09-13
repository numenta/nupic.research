#!/bin/zsh

# CSV files below will be updated sequentially, then a markdown table gets written out
# after results for the last model in a group (like base or small) are added

# Bert base and bert base trifecta
SMALL_CSV=~/nta/nupic.research/projects/transformers/results/small_condensed_finetuning_results.csv
SMALL_MD=~/nta/nupic.research/projects/transformers/results/small_condensed_finetuning_results.md

rm ${SMALL_CSV}

## Small bert

# Note, many of the hyperparameter chases failed for an unknown reason
# after finetuing on 7/9 tasks. Thus, there are multiple files passed in
# for hp chase models with names like "first two" or "follow up" to indicate
# additional runs to get any remaining results

# Bert small 100k
# Note, these have the correct batch size, unlike previous small_bert
python ../export_finetuning_results.py \
                ~/nta/finetuning/finetuning_small_bert_large_dataset_100k_glue \
                ~/nta/finetuning/small_bert_big_dataset_hp_chase \
                --model_name Dense_small \
                --csv ${SMALL_CSV} \
                --pretrained_model Dense_small

# Trifecta small 80
python ../export_finetuning_results.py \
                ~/nta/finetuning/finetuning_small_bert_trifecta_100k_glue \
                ~/nta/finetuning/trifecta_80_small_hp_chase \
                ~/nta/finetuning/trifecta_80_small_hp_chase_first_two \
                --model_name Trifecta_small_80 \
                --csv ${SMALL_CSV} \
                --pretrained_model Trifecta_small_80

# Trifecta small 85
python ../export_finetuning_results.py \
                ~/nta/finetuning/finetuning_small_bert_sparse_85_trifecta_100k_glue \
                ~/nta/finetuning/trifecta_85_small_hp_chase \
                ~/nta/finetuning/trifecta_85_small_hp_chase_first_two \
                --model_name Trifecta_small_85 \
                --csv ${SMALL_CSV} \
                --pretrained_model Trifecta_small_85

# Trifecta small 90
python ../export_finetuning_results.py \
                ~/nta/finetuning/finetuning_small_bert_sparse_90_trifecta_100k_glue \
                ~/nta/finetuning/trifecta_90_small_hp_chase \
                ~/nta/finetuning/trifecta_90_small_hp_chase_first_two \
                --model_name Trifecta_small_90 \
                --csv ${SMALL_CSV} \
                --pretrained_model Trifecta_small_90

# Trifecta small 2x
python ../export_finetuning_results.py \
                ~/nta/finetuning/finetuning_small_bert_sparse_2x_trifecta_100k_glue \
                ~/nta/finetuning/trifecta_2x_small_hp_chase \
                ~/nta/finetuning/trifecta_2x_small_hp_chase_follow_up \
                --model_name Trifecta_small_2x \
                --csv ${SMALL_CSV} \
                --pretrained_model Trifecta_small_2x

# Trifecta small 4x
python ../export_finetuning_results.py \
                ~/nta/finetuning/finetuning_small_bert_sparse_4x_trifecta_100k_glue \
                ~/nta/finetuning/trifecta_4x_small_hp_chase \
                ~/nta/finetuning/trifecta_4x_small_hp_chase_first_two \
                --model_name Trifecta_small_4x \
                --csv ${SMALL_CSV} \
                --pretrained_model Trifecta_small_4x \
                 --md ${SMALL_MD}
