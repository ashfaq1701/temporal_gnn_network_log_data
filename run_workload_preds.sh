#!/bin/bash

# Disable Python buffering
export PYTHONUNBUFFERED=1

# Check if microservice_id is passed as a parameter
if [ -z "$1" ]; then
  echo "Usage: $0 <microservice_id> [base_dir] [n_runs] [d_model] [embedding_scaling_type] [embedding_scaling_factor] [test_microservice_id] [should_reverse_data]"
  exit 1
fi

# Assign the microservice_id from the command-line argument
microservice_id=$1

# Assign base_dir from the second command-line argument or use default value
base_dir=${2:-"/its/home/ms2420/alibaba_gnn_results"}

# Assign n_runs from the third command-line argument or use default value of 10
n_runs=${3:-10}

# Assign d_model from the fourth command-line argument or use default value of 1024
d_model=${4:-1024}
double_d_model=$((d_model * 2))

embedding_scaling_type=${5:-"max"}

embedding_scaling_factor=${6:-1.0}

test_microservice_id=${7:-"none"}

should_reverse_data=${8:-"none"}

# Run the workload prediction tasks with different configurations
python3 main.py --task predict_workload --d_model $d_model --only_use_target_microservice --microservice_id $microservice_id --patience 5 --n_runs $n_runs --output_dir "$base_dir" --embedding_scaling_type "$embedding_scaling_type" --embedding_scaling_factor $embedding_scaling_factor --test_microservice_id "$test_microservice_id" --should_reverse_data "$should_reverse_data"
python3 main.py --task predict_workload --d_model $d_model --only_use_target_microservice --ignore_temporal_embedding --microservice_id $microservice_id --patience 5 --n_runs $n_runs --output_dir "$base_dir" --embedding_scaling_type "$embedding_scaling_type" --embedding_scaling_factor $embedding_scaling_factor --test_microservice_id "$test_microservice_id" --should_reverse_data "$should_reverse_data"
python3 main.py --task predict_workload --d_model $d_model --only_use_target_microservice --seq_len 48 --label_len 24 --pred_len 12 --microservice_id $microservice_id --patience 5 --n_runs $n_runs --output_dir "$base_dir" --embedding_scaling_type "$embedding_scaling_type" --embedding_scaling_factor $embedding_scaling_factor --test_microservice_id "$test_microservice_id" --should_reverse_data "$should_reverse_data"
python3 main.py --task predict_workload --d_model $d_model --only_use_target_microservice --ignore_temporal_embedding --seq_len 48 --label_len 24 --pred_len 12 --microservice_id $microservice_id --patience 5 --n_runs $n_runs --output_dir "$base_dir" --embedding_scaling_type "$embedding_scaling_type" --embedding_scaling_factor $embedding_scaling_factor --test_microservice_id "$test_microservice_id" --should_reverse_data "$should_reverse_data"
python3 main.py --task predict_workload --d_model $d_model --only_use_target_microservice --seq_len 96 --label_len 48 --pred_len 24 --microservice_id $microservice_id --patience 5 --n_runs $n_runs --output_dir "$base_dir" --embedding_scaling_type "$embedding_scaling_type" --embedding_scaling_factor $embedding_scaling_factor --test_microservice_id "$test_microservice_id" --should_reverse_data "$should_reverse_data"
python3 main.py --task predict_workload --d_model $d_model --only_use_target_microservice --ignore_temporal_embedding --seq_len 96 --label_len 48 --pred_len 24 --microservice_id $microservice_id --patience 5 --n_runs $n_runs --output_dir "$base_dir" --embedding_scaling_type "$embedding_scaling_type" --embedding_scaling_factor $embedding_scaling_factor --test_microservice_id "$test_microservice_id" --should_reverse_data "$should_reverse_data"
python3 main.py --task predict_workload --d_model ${double_d_model} --only_use_target_microservice --seq_len 240 --label_len 120 --pred_len 60 --microservice_id $microservice_id --patience 5 --n_runs $n_runs --output_dir "$base_dir" --embedding_scaling_type "$embedding_scaling_type" --embedding_scaling_factor $embedding_scaling_factor --test_microservice_id "$test_microservice_id" --should_reverse_data "$should_reverse_data"
python3 main.py --task predict_workload --d_model ${double_d_model} --only_use_target_microservice --ignore_temporal_embedding --seq_len 240 --label_len 120 --pred_len 60 --microservice_id $microservice_id --patience 5 --n_runs $n_runs --output_dir "$base_dir" --embedding_scaling_type "$embedding_scaling_type" --embedding_scaling_factor $embedding_scaling_factor --test_microservice_id "$test_microservice_id" --should_reverse_data "$should_reverse_data"
python3 main.py --task predict_workload --d_model ${double_d_model} --microservice_id $microservice_id --patience 5 --n_runs $n_runs --output_dir "$base_dir" --embedding_scaling_type "$embedding_scaling_type" --embedding_scaling_factor $embedding_scaling_factor --test_microservice_id "$test_microservice_id" --should_reverse_data "$should_reverse_data"
python3 main.py --task predict_workload --d_model ${double_d_model} --ignore_temporal_embedding --microservice_id $microservice_id --patience 5 --n_runs $n_runs --output_dir "$base_dir" --embedding_scaling_type "$embedding_scaling_type" --embedding_scaling_factor $embedding_scaling_factor --test_microservice_id "$test_microservice_id" --should_reverse_data "$should_reverse_data"
