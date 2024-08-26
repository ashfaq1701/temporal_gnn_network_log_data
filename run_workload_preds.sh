#!/bin/bash

# Disable Python buffering
export PYTHONUNBUFFERED=1

# Check if microservice_id is passed as a parameter
if [ -z "$1" ]; then
  echo "Usage: $0 <microservice_id> [n_runs]"
  exit 1
fi

# Assign the microservice_id from the command-line argument
microservice_id=$1

# Assign n_runs from the second command-line argument or use default value of 10
n_runs=${2:-10}

# Define the base directory
base_dir="/its/home/ms2420/alibaba_gnn_results/workload_prediction_results_$microservice_id"

# Remove the old results directory if it exists and create a new one
rm -rf "$base_dir"
mkdir -p "$base_dir"

# Run the workload prediction tasks with different configurations
python3 main.py --task predict_workload --d_model 2048 --n_heads 16 --e_layers 4 --d_layers 2 --only_use_target_microservice --microservice_id $microservice_id --patience 5 --n_runs $n_runs --output_dir "$base_dir/univariate_with_embedding/"
python3 main.py --task predict_workload --d_model 2048 --n_heads 16 --e_layers 4 --d_layers 2 --only_use_target_microservice --ignore_temporal_embedding --microservice_id $microservice_id --patience 5 --n_runs $n_runs --output_dir "$base_dir/univariate_without_embedding/"
python3 main.py --task predict_workload --d_model 2048 --n_heads 16 --e_layers 4 --d_layers 2 --only_use_target_microservice --seq_len 48 --label_len 24 --pred_len 12 --microservice_id $microservice_id --patience 5 --n_runs $n_runs --output_dir "$base_dir/univariate_with_embedding_long_range_pred/"
python3 main.py --task predict_workload --d_model 2048 --n_heads 16 --e_layers 4 --d_layers 2 --only_use_target_microservice --ignore_temporal_embedding --seq_len 48 --label_len 24 --pred_len 12 --microservice_id $microservice_id --patience 5 --n_runs $n_runs --output_dir "$base_dir/univariate_without_embedding_long_range_pred/"
python3 main.py --task predict_workload --d_model 2048 --n_heads 16 --e_layers 4 --d_layers 2 --only_use_target_microservice --seq_len 96 --label_len 48 --pred_len 24 --microservice_id $microservice_id --patience 5 --n_runs $n_runs --output_dir "$base_dir/univariate_with_embedding_long_range_pred_24/"
python3 main.py --task predict_workload --d_model 2048 --n_heads 16 --e_layers 4 --d_layers 2 --only_use_target_microservice --ignore_temporal_embedding --seq_len 96 --label_len 48 --pred_len 24 --microservice_id $microservice_id --patience 5 --n_runs $n_runs --output_dir "$base_dir/univariate_without_embedding_long_range_pred_24/"
python3 main.py --task predict_workload --d_model 2048 --n_heads 16 --e_layers 4 --d_layers 2 --only_use_target_microservice --seq_len 240 --label_len 120 --pred_len 60 --microservice_id $microservice_id --patience 5 --n_runs $n_runs --output_dir "$base_dir/univariate_with_embedding_long_range_pred_60/"
python3 main.py --task predict_workload --d_model 2048 --n_heads 16 --e_layers 4 --d_layers 2 --only_use_target_microservice --ignore_temporal_embedding --seq_len 240 --label_len 120 --pred_len 60 --microservice_id $microservice_id --patience 5 --n_runs $n_runs --output_dir "$base_dir/univariate_without_embedding_long_range_pred_60/"
python3 main.py --task predict_workload --d_model 2048 --n_heads 16 --e_layers 4 --d_layers 2 --microservice_id $microservice_id --patience 5 --n_runs $n_runs --output_dir "$base_dir/multivariate_with_embedding/"
python3 main.py --task predict_workload --d_model 2048 --n_heads 16 --e_layers 4 --d_layers 2 --ignore_temporal_embedding --microservice_id $microservice_id --patience 5 --n_runs $n_runs --output_dir "$base_dir/multivariate_without_embedding/"
