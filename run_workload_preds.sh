#!/bin/bash

# Disable Python buffering
export PYTHONUNBUFFERED=1

# Check if microservice_id is passed as a parameter
if [ -z "$1" ]; then
  echo "Usage: $0 <microservice_id> [base_dir] [n_runs] [d_model] [embedding_scaling_type] [embedding_scaling_factor]"
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

embedding_scaling_type=${5:-"max"}

embedding_scaling_factor=${6:-1.0}

# Define the full directory path
full_dir="$base_dir/workload_prediction_results_$microservice_id"

# Remove the old results directory if it exists and create a new one
rm -rf "$full_dir"
mkdir -p "$full_dir"

# Run the workload prediction tasks with different configurations
python3 main.py --task predict_workload --d_model $d_model --only_use_target_microservice --microservice_id $microservice_id --patience 5 --n_runs $n_runs --output_dir "$full_dir/univariate_with_embedding/" --embedding_scaling_type "$embedding_scaling_type" --embedding_scaling_factor $embedding_scaling_factor
python3 main.py --task predict_workload --d_model $d_model --only_use_target_microservice --ignore_temporal_embedding --microservice_id $microservice_id --patience 5 --n_runs $n_runs --output_dir "$full_dir/univariate_without_embedding/" --embedding_scaling_type "$embedding_scaling_type" --embedding_scaling_factor $embedding_scaling_factor
python3 main.py --task predict_workload --d_model $d_model --only_use_target_microservice --seq_len 48 --label_len 24 --pred_len 12 --microservice_id $microservice_id --patience 5 --n_runs $n_runs --output_dir "$full_dir/univariate_with_embedding_long_range_pred/" --embedding_scaling_type "$embedding_scaling_type" --embedding_scaling_factor $embedding_scaling_factor
python3 main.py --task predict_workload --d_model $d_model --only_use_target_microservice --ignore_temporal_embedding --seq_len 48 --label_len 24 --pred_len 12 --microservice_id $microservice_id --patience 5 --n_runs $n_runs --output_dir "$full_dir/univariate_without_embedding_long_range_pred/" --embedding_scaling_type "$embedding_scaling_type" --embedding_scaling_factor $embedding_scaling_factor
python3 main.py --task predict_workload --d_model $d_model --only_use_target_microservice --seq_len 96 --label_len 48 --pred_len 24 --microservice_id $microservice_id --patience 5 --n_runs $n_runs --output_dir "$full_dir/univariate_with_embedding_long_range_pred_24/" --embedding_scaling_type "$embedding_scaling_type" --embedding_scaling_factor $embedding_scaling_factor
python3 main.py --task predict_workload --d_model $d_model --only_use_target_microservice --ignore_temporal_embedding --seq_len 96 --label_len 48 --pred_len 24 --microservice_id $microservice_id --patience 5 --n_runs $n_runs --output_dir "$full_dir/univariate_without_embedding_long_range_pred_24/" --embedding_scaling_type "$embedding_scaling_type" --embedding_scaling_factor $embedding_scaling_factor
python3 main.py --task predict_workload --d_model $d_model --n_heads 16 --e_layers 3 --d_layers 2 --s_layers '4,3,2' --dropout 0.1 --only_use_target_microservice --seq_len 240 --label_len 120 --pred_len 60 --microservice_id $microservice_id --patience 5 --n_runs $n_runs --output_dir "$full_dir/univariate_with_embedding_long_range_pred_60/" --embedding_scaling_type "$embedding_scaling_type" --embedding_scaling_factor $embedding_scaling_factor
python3 main.py --task predict_workload --d_model $d_model --n_heads 16 --e_layers 3 --d_layers 2 --s_layers '4,3,2' --dropout 0.1 --only_use_target_microservice --ignore_temporal_embedding --seq_len 240 --label_len 120 --pred_len 60 --microservice_id $microservice_id --patience 5 --n_runs $n_runs --output_dir "$full_dir/univariate_without_embedding_long_range_pred_60/" --embedding_scaling_type "$embedding_scaling_type" --embedding_scaling_factor $embedding_scaling_factor
python3 main.py --task predict_workload --d_model 2048 --microservice_id $microservice_id --patience 5 --n_runs $n_runs --output_dir "$full_dir/multivariate_with_embedding/" --embedding_scaling_type "$embedding_scaling_type" --embedding_scaling_factor $embedding_scaling_factor
python3 main.py --task predict_workload --d_model 2048 --ignore_temporal_embedding --microservice_id $microservice_id --patience 5 --n_runs $n_runs --output_dir "$full_dir/multivariate_without_embedding/" --embedding_scaling_type "$embedding_scaling_type" --embedding_scaling_factor $embedding_scaling_factor
