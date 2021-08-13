#!/bin/zsh

# Bert base and bert base trifecta
BASE_CSV=~/nta/nupic.research/projects/transformers/results/base_finetuning_results.csv
BASE_MD=~/nta/nupic.research/projects/transformers/results/base_finetuning_results.md

# Bert small and bert small trifecta
SMALL_CSV=~/nta/nupic.research/projects/transformers/results/small_finetuning_results.csv
SMALL_MD=~/nta/nupic.research/projects/transformers/results/small_finetuning_results.md

rm ${BASE_CSV}
rm ${SMALL_CSV}

## Base bert

# Bert 100k
python ../export_finetuning_results.py ~/nta/tmp/finetuning/finetuning_bert100k_glue_simple_no_ESC \
            --model_name bert_100k_simple --csv ${BASE_CSV} --pretrained_model Dense
python ../export_finetuning_results.py ~/nta/finetuning/bert_100k_hp_chase \
            --model_name bert_100k_hp_chase --csv ${BASE_CSV} --pretrained_model Dense

# Trifecta: 80
python ../export_finetuning_results.py ~/nta/finetuning/finetuning_bert_sparse_trifecta_100k_glue_get_info_nb \
            --model_name trifecta_80_100k_glue_get_info --csv ${BASE_CSV} --pretrained_model Trifecta_80
python ../export_finetuning_results.py ~/nta/finetuning/trifecta_80_hp_chase \
            --model_name trifecta_80_glue_hp_chase --csv ${BASE_CSV} --pretrained_model Trifecta_80

# Trifecta 85
python ../export_finetuning_results.py ~/nta/finetuning/finetuning_bert_sparse_85_trifecta_100k_glue_get_info \
            --model_name trifecta_85_glue_get_info --csv ${BASE_CSV} --pretrained_model Trifecta_85
python ../export_finetuning_results.py ~/nta/finetuning/trifecta_85_hp_chase \
            --model_name trifecta_85_glue_hp_chase --csv ${BASE_CSV} --pretrained_model Trifecta_85

# Trifecta 90
python ../export_finetuning_results.py ~/nta/finetuning/finetuning_bert_sparse_90_trifecta_100k_glue_get_info \
            --model_name trifecta_90_glue_get_info --csv ${BASE_CSV} --pretrained_model Trifecta_90
python ../export_finetuning_results.py ~/nta/finetuning/trifecta_90_hp_chase ~/nta/finetuning/trifecta_90_hp_chase_mnli \
            --model_name trifecta_90_glue_hp_chase --csv ${BASE_CSV} --pretrained_model Trifecta_90

# Trifecta 2x
python ../export_finetuning_results.py ~/nta/finetuning/finetuning_bert_sparse_trifecta_2x_get_info \
            --model_name trifecta_2x_glue_get_info --csv ${BASE_CSV} --pretrained_model Trifecta_2x \
            --md ${BASE_MD}


## Small bert

# Bert small 100k: 
python ../export_finetuning_results.py ~/nta/finetuning/finetuning_small_bert_100k_glue \
            --model_name small_bert_100k_get_info --csv ${SMALL_CSV} --pretrained_model Dense_small

# Trifecta small: 80, 85, 90, 2x, 4x
python ../export_finetuning_results.py ~/nta/finetuning/finetuning_small_bert_trifecta_100k_glue \
        --model_name trifecta_small --csv ${SMALL_CSV} --pretrained_model Trifecta_small_80
python ../export_finetuning_results.py ~/nta/finetuning/finetuning_small_bert_sparse_85_trifecta_100k_glue \
            --model_name trifecta_small_85_get_info --csv ${SMALL_CSV} --pretrained_model Trifecta_small_85
python ../export_finetuning_results.py ~/nta/finetuning/finetuning_small_bert_sparse_90_trifecta_100k_glue \
            --model_name trifecta_small_90_get_info --csv ${SMALL_CSV} --pretrained_model Trifecta_small_90
python ../export_finetuning_results.py ~/nta/finetuning/finetuning_small_bert_sparse_2x_trifecta_100k_glue \
            --model_name trifecta_small_2x_get_info --csv ${SMALL_CSV} --pretrained_model Trifecta_small_2x
python ../export_finetuning_results.py ~/nta/finetuning/finetuning_small_bert_sparse_4x_trifecta_100k_glue \
            --model_name trifecta_small_4x_get_info --csv ${SMALL_CSV} --pretrained_model Trifecta_small_4x \
            --md ${SMALL_MD}
