# Transformers

## Results Bert

In progress, current results:

|            | average_bert     | average_glue      | cola           | mnli                         | mrpc        | qnli     | qqp         | rte      | sst2     | stsb                  | wnli     |
|:-----------|:-----------------|:------------------|:---------------|:-----------------------------|:------------|:---------|:------------|:---------|:---------|:----------------------|:---------|
|            | Average w/o wnli | Average all tasks | Matthew's corr | Matched acc./Mismatched acc. | F1/Accuracy | Accuracy | Accuracy/F1 | Accuracy | Accuracy | Person/Spearman corr. | Accuracy |
| bert_HF    | 81.67            | 78.85             | 56.53          | 83.91/84.10                  | 88.85/84.07 | 90.66    | 90.71/87.49 | 65.70    | 92.32    | 88.64/88.48           | 56.34    |
| bert_paper|          79.60 |             -  |  52.10 | 84.60/83.40          | 88.90/- | 90.50     | 71.20/-     | 66.40     | 93.50     | 85.80        |      - |
| bert_1mi |          78.81 |          74.22 |  45.36 | 79.51/80.58 | 88.63/84.64 |  91.67 | 90.82/87.60 | 58.59 |   91.90 | 87.13/87.00 |   37.50 |
| bert_700k |          79.44 |          74.25 |  47.33 | 83.43/83.67 | 88.93/84.38 |  91.75 | 90.38/86.94 |  58.20 |  92.48 | 86.88/86.85 |  32.81 |
| bert_100k |          74.45 |      68.07 |  40.12 |    78.89/-   | 83.39/75.52 |  88.05 | 89.10/85.31 |  54.30 |  88.08 | 83.99/83.82 |  23.44 |

</br>
</br>

Model description:
* bert_HF: reported by HuggingFace, see https://github.com/huggingface/transformers/tree/master/examples/text-classification. No inner loop of hyperparameter optimization for each finetuning task. Pretrained for 1mi steps of batch size 256.
* bert_paper: reported in BERT paper, see https://arxiv.org/abs/1810.04805. Hyperparameters for each task are chosen in a grid search with 18 different variations per task, and best results reported. Pretrained for 1mi steps of batch size 256.
* bert_Nk: where N is the number of steps in pretraining. These refer to models trained by ourselves for less than or more than 1mi steps. They might be intermediate checkpoints of longer runs or a shorter run executed to completion. Batch size is 256 unless specified otherwise in the model name. No inner loop of hyperparameter optimization for each finetuning task, follows same finetuning hyperparameters defined by bert_HF. Results for cola, rte and wnli are a max over two runs, remaining tasks are single runs.

Known issues, to be investigated/fixed:
* A lot of variance in finetuning, specially in small datasets, like rte and wnli, and highly unbalanced datasets such as cola. In cola for example, same model will output results from 14 to 47 using exact similar finetuning regimes. Maxing over several runs helps in stabilizing finetuning results, making it easier to compare across different models.
* We don't have 100% coverage of validation set. Some samples of the validation set are given the label "-100", most likely for the tokenizer not being able to process them. These samples are excluded for validation. That coverage varies from 95-100%, with most tasks being close to 100%, but cola and mnli specifically have ~95% coverage. Not clear whether the coverage for bert_paper and bert_hf is 100%.
* Only GLUE tasks included in the table. GLUE covers sequence classification or other tasks such as Q&A or multiple choice reframed as sequence classification. BERT also reports results on SQuaD (question answering), SWAG (multiple choice) and CoNLL (named entity recognition).
* To add to the table perplexity of the original pretrained models.

> Finetuning is important to evaluate a language model effectiveness when being used for transfer learning in downstream tasks. However, it is not a reliable metric to use for optimization. These numbers will vary with a significant margin of error between two different runs. Use perplexity instead to guide hyperparameter search and development of language models.

## How to run - single node

Local implementation using Huggingface. To run, create a new experiment dict under experiments and run using:

`python run.py <experiment_name>`

You can also run multiple experiments in sequence:

`python run.py <experiment_name_A>  <experiment_name_B>`

Other accepted ways are passing arguments directly in command line or through a json file. See run.py for more details.

## How to run - multiple nodes

To run it in multiple nodes, you will require the run script from transformers-cli-utils (it can be found under ray folder in the infrastructure repository). Make sure to add transformers-cli-utils root folder to PATH after downloading.

First define the location of your cluster yaml file in `RAY_CONFIG_FILE` and the location of your AWS certification file (.pem file) in `AWS_CERT_FILE`:

`export RAY_CONFIG_FILE=<path to ray config file>`

`export AWS_CERT_FILE=<path to your AWS certification file>`

The easiest way is to add the commands that create new environment variables or modify existing ones (like PATH) to your ~/.bash_profile, so they are automatically initialized every time you open a new bash terminal.

The head and worker nodes should be the of the same type of instance, and the type selected should contain at least one GPU.
Set the variable `initial_workers` in the yaml file to initialize them all along with the head node.
Then initialize your cluster:

`ray up <path to ray config file>`

After the head and worker nodes are initialized, run using the bash script provided in transformers-cli-utils:

`run.sh <experiment_name>`

As in single node, you can run multiple experiments in sequence. When using the script with multiple experiments, wrap the experiment names in quotes so it is read as a single argument:

`run.sh "<experiment_name_A> <experiment_name_B>"`

Wait a few minutes and the output of all instances will be redirected to your local terminal.
You can follow up the experiments in the wandb link shown when training starts.

If you need to resync the files after a local change, use the sync script available in transformers-cli-utils:

`sync.sh <path to file or folder local> <path to file or folder remote>`

For any additional commands, use the remote script. For example, to verify GPU usage on all nodes, do:

`remote.sh nvidia-smi`

Or to kill all running python processes:

`remote.sh "pkill -f python"`

If required, you can reboot all instances by selecting all of them in EC2 console and selecting action > reboot from the instance drop-down menu.

### Running from the head node

There is also an option of running the multiple nodes scripts from a ray head node instead of local. In this case, the head node will play the part of local and only the worker nodes will run the commands. For this scenario it is advised to have a simple non-GPU head node, since it will only be used to issue commands to the workers.

After accessing the head node via ssh or attaching a screen, run an experiment using the same command described above: `run.sh <experiment_name>`. See transformers-cli-utils readme for more information on how to use and what modifications are required to the yaml file.

## Installing

You will require the libraries `datasets` and `transformers`.
* `datasets` can be installed using pip or from source
* install `transformers` from source, by cloning and running `pip install -e .`

The `requirement.txt` file contains a specific SHA if you want to reproduce a tested environment. We are using the latest features from these libraries and will incorporate others which are soon to be released, so for the moment those might change at a fast pace. Once we have the need to establish reproducible results we should consider more stable requirements.

## Additional notes

`transformers-cli-utils` is in a private repository. If you woud like to use it, feel free to drop me an email at lsouza at numenta dot com.