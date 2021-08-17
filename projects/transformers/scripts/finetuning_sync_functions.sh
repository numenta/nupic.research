### Warning, these are one-off functions I've been using to sync results. No quality assurance...at all...

# Before using these functions, set up two data directories for youself. 
# One for finetuning runs.
# One for hyperparameter search runs.
# Also replace /Users/bcohen/... with the appropriate data directory.

# Sync all task_results.p files from finetuning experiments.
# 1: ray config
# 2: file with list of finetuning configs, one per line, with an empty line at the end.
sync_finetuning () {
    while IFS= read -r line; do
	    mkdir -p /Users/bcohen/nta/finetuning/${line}/
        ray rsync_down ${1} "/home/ec2-user/nta/results/experiments/transformers/${line}/task_results.p" /Users/bcohen/nta/finetuning/${line}/task_results.p
    done < ${2}
}

# Sync down files that can be uploaded to the glue leaderboard
# 1: ray config
# 2: file with list of finetuning configs, same format as above
sync_finetuning_test () {
    declare -a tasks=("CoLA" "SST-2" "MRPC" "STS-B" "QQP" "MNLI-M" "MNLI-MM" "QNLI" "RTE" "WNLI")
    mkdir -p /Users/bcohen/nta/finetuning/${2}/
    for task in tasks
    do
        ray rsync_down ${1} "/home/ec2-user/nta/results/experiments/transformers/${2}/${task}_best.tsv" /Users/bcohen/nta/finetuning/${2}/${task}.tsv
    done
}

# Wrap up a set of files and submit the zip file to the Glue leaderboard
zip_glue () {
    zip -r submission.zip ./*.tsv
}

# Sync the results of hyperparameter searches. It assumes all 9 tasks have data and will try to sync each one.
# If you see a lot of errors, it might be because you are trying to sync all 9 tasks, but not all 9 are there. 
# This is fine as ray will fail safely, and then you will rm any empty directories at the end.
# 1 ray config
# 2 file with config names, one per line, must have trailing empty line, no trailing whitespace
sync_hp_search () {
    declare -a tasks=("cola" "sst2" "mrpc" "stsb" "qqp" "mnli" "qnli" "rte" "wnli")
    while IFS= read -r line; do
        for task in ${tasks}; do
            mkdir -p /Users/bcohen/nta/hp_search/${line}/${task}/
            ray rsync_down ${1} "/home/ec2-user/nta/results/experiments/transformers/${line}/${task}/_obj*" /Users/bcohen/nta/hp_search/${line}/${task}/
        done
        # delete any empty directories
        find /Users/bcohen/nta/hp_search/${line}/ -type d -empty -delete
    done < ${2}
}


# A wrapper for sync_hp_search and analyze_all_hp_search.sh
sync_and_export_hp () {
    sync_hp_search ${1} ~/nta/hp_search/configs.txt
    ~/nta/nupic.research/projects/transformers/scripts/analyze_all_hp_search.sh
}