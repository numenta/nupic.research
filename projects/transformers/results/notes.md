**Note:** `sparse_80%_kd_onecycle_lr_rigl` is not actually 80% sparse. It's just under at 79.58%. This is because the token embeddings are fully dense. Future work will either sparsify these, or make the other layers more sparse to achieve the full 80%.


</br>
</br>

Model description:
* bert_HF: reported by HuggingFace, see https://github.com/huggingface/transformers/tree/master/examples/text-classification. No inner loop of hyperparameter optimization for each finetuning task. Pretrained for 1mi steps of batch size 256.
* bert_paper: reported in BERT paper, see https://arxiv.org/abs/1810.04805. Hyperparameters for each task are chosen in a grid search with 18 different variations per task, and best results reported. Pretrained for 1mi steps of batch size 256.
* bert_Nk: where N is the number of steps in pretraining. These refer to models trained by ourselves for less than or more than 1mi steps. They might be intermediate checkpoints of longer runs or a shorter run executed to completion. Batch size is 256 unless specified otherwise in the model name. No inner loop of hyperparameter optimization for each finetuning task, follows same finetuning hyperparameters defined by bert_HF. Results for {cola, rte, wnli} are a max over 10 runs, {sstb, mrpc} are a max over 3 runs, and the remaining tasks are single runs.

Known issues, to be investigated/fixed:
* A lot of variance in finetuning, specially in small datasets, like rte and wnli, and highly unbalanced datasets such as cola. In cola for example, same model will output results from 14 to 47 using exact similar finetuning regimes. Maxing over several runs helps in stabilizing finetuning results, making it easier to compare across different models.f
* Only GLUE tasks included in the table. GLUE covers sequence classification or other tasks such as Q&A or multiple choice reframed as sequence classification. BERT also reports results on SQuaD (question answering), SWAG (multiple choice) and CoNLL (named entity recognition).
* To add to the table perplexity of the original pretrained models.

> Finetuning is important to evaluate a language model effectiveness when being used for transfer learning in downstream tasks. However, it is not a reliable metric to use for optimization. These numbers will vary with a significant margin of error between two different runs. Use perplexity instead to guide hyperparameter search and development of language models.

Known Bugs
* Finetuning occasionally stalls and needs restarting. This may be related to [this](https://github.com/huggingface/transformers/issues/5486) documented issue. Although this deadlock doesn't happen every time, the only way found to entirely avoid this is to disable wandb logging.