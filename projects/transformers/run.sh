#!/bin/bash

# Accepts two arguments:
#   1 - number of the cluster. Expect cluster files to have following path:
#       ~/nta/ray_config/ray_user<cluster_number>.yaml
#   2 - experiment name as defined in one of the configs

# ------ Sync latest to experiment -------------

# TODO: sync files with all the nodes

# ------ Get head private IP -------------

get_head_private_ip() {
    local ip=$(ssh -o "StrictHostKeyChecking no" -i \
              ~/.ssh/ray-autoscaler_us-west-2.pem \
              ec2-user@$(ray get-head-ip ~/nta/ray_config/ray_user$1.yaml) \
              hostname -I | awk '{print $1}')
    echo $ip
}

head_private_ip=$(get_head_private_ip $1)
echo "Head private IP: $head_private_ip"

# ------ Get head and workers public IP -------------

head_public_ip=$(ray get-head-ip ~/nta/ray_config/ray_user$1.yaml)
echo "Head public IP: $head_public_ip"

get_worker_public_ips() {
    local counter=0
    for ip in $(ray get-worker-ips ~/nta/ray_config/ray_user$1.yaml)
    do
        (( counter++ ))
        echo $ip
    done
}

worker_public_ips=$(get_worker_public_ips $1)
echo "Worker public IPs: ${worker_public_ips}"

# ------ Get number of workers -------------

get_num_workers() {
    local max_workers=$(cat ~/nta/ray_config/ray_user$1.yaml | grep max_workers)
    echo ${max_workers: -1}
}

num_workers=$(get_num_workers $1)
echo "Number of (max) workers: $num_workers"

# ------ Get number of GPUs -------------

get_num_gpus() {
    local instance_type=$(cat ~/nta/ray_config/ray_user$1.yaml | grep "  InstanceType" | sed -n '2p' | awk '{print $2}')
    case $instance_type in
        p3.2xlarge)
            echo 1
            ;;
        p3.8xlarge)
            echo 4
            ;;
        p3.16xlarge)
            echo 8
            ;;
        p3dn.24xlarge)
            echo 8
            ;;
        *)
            echo 0
            ;;
    esac
}

num_gpus=$(get_num_gpus $1)
echo "Number of GPUS per instance: $num_gpus"

# ------ Generate a random port number -------------
# Port number in interval [1200, 1300].
# TODO: verify if port is available, if not select another

random_port=$(( $RANDOM % 100 + 1200 ))
echo "Random port selected: $random_port"

# ------ Run file -------------

run_file() {

    local counter=0
    local all_ips="${head_public_ip} ${worker_public_ips}"
    for ip in $all_ips
    do
        echo "Running command in instance $counter : $ip"

        ssh -o "StrictHostKeyChecking no" -i \
        ~/.ssh/ray-autoscaler_us-west-2.pem \
        ec2-user@$ip \
        python -m torch.distributed.launch \
            --nproc_per_node $num_gpus \
            --nnodes $(( num_workers+1 )) \
            --node_rank $counter \
            --master_addr $head_private_ip \
            --master_port $random_port \
            "~/nta/nupic.research/projects/transformers/run.py" \
            --experiment $1 &

        (( counter++ ))
    done

}

run_file $2

# find a way to get the process id and kill all of them if I need it
