#!/bin/zsh

# 
CSV=~/nta/nupic.research/projects/transformers/results/finetuning_results.csv
MD=~/nta/nupic.research/projects/transformers/results/finetuning_results.md
rm ${CSV}

python export_finetuning_results.py ~/nta/tmp/finetuning/finetuning_bert100k_glue_simple_no_ESC --model_name bert_100k_simple_no_esc_unsparsified --csv ${CSV}
python export_finetuning_results.py ~/nta/tmp/finetuning/finetuning_bert_sparse_trifecta_100k_glue_simple_no_ESC --model_name trifecta_80_simple_no_esc_unsparsified --csv ${CSV} 

python export_finetuning_results.py ~/nta/tmp/finetuning/finetuning_bert_sparse_80_trifecta_100k_glue --model_name trifecta_80_glue_unsparsified --csv ${CSV}
python export_finetuning_results.py ~/nta/tmp/finetuning/finetuning_bert_sparse_85_trifecta_100k_glue --model_name trifecta_85_glue_unsparsified --csv ${CSV}
python export_finetuning_results.py ~/nta/tmp/finetuning/finetuning_bert_sparse_90_trifecta_100k_glue --model_name trifecta_90_glue_unsparsified --csv ${CSV}

python export_finetuning_results.py ~/nta/finetuning/finetuning_bert_sparse_trifecta_100k_glue --model_name trifecta_80_glue_nlb_not_unsparsify --csv ${CSV}
python export_finetuning_results.py ~/nta/finetuning/finetuning_bert_sparse_85_trifecta_100k_glue --model_name trifecta_85_glue_nlb_not_unsparsify --csv ${CSV}
python export_finetuning_results.py ~/nta/finetuning/finetuning_bert_sparse_90_trifecta_100k_glue --model_name trifecta_90_glue_nlb_not_unsparsify --csv ${CSV} 

python export_finetuning_results.py ~/nta/finetuning/finetuning_bert_sparse_trifecta_100k_glue_get_info --model_name trifecta_80_glue_get_info --csv ${CSV}
python export_finetuning_results.py ~/nta/finetuning/finetuning_bert_sparse_85_trifecta_100k_glue_get_info --model_name trifecta_85_glue_get_info --csv ${CSV}
python export_finetuning_results.py ~/nta/finetuning/finetuning_bert_sparse_90_trifecta_100k_glue_get_info --model_name trifecta_90_glue_get_info --csv ${CSV}

python export_finetuning_results.py ~/nta/finetuning/finetuning_bert_sparse_trifecta_100k_glue_get_info_MC_direct --model_name trifecta_80_glue_get_info_MC_direct --csv ${CSV}
python export_finetuning_results.py ~/nta/finetuning/finetuning_bert_sparse_trifecta_100k_glue_get_info_bug_fixed --model_name trifecta_80_glue_get_info_bug_fixed --csv ${CSV} 
python export_finetuning_results.py ~/nta/finetuning/finetuning_bert_sparse_trifecta_100k_glue_get_info_bug_fixed_again --model_name trifecta_80_glue_get_info_bug_fixed_again --csv ${CSV}
python export_finetuning_results.py ~/nta/finetuning/finetuning_bert_sparse_trifecta_100k_glue_get_info_nb --model_name trifecta_80_100k_glue_get_info_nb --csv ${CSV} --md ${MD}

