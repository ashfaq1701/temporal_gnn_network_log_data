#!/bin/bash

# Disable Python buffering
export PYTHONUNBUFFERED=1

# Check if microservice_id is passed as a parameter
if [ -z "$1" ]; then
  echo "Usage: $0 <model_base_path> <output_base_dir> <test_microservice_id> <train_microservice_id> [should_reverse_data]"
  exit 1
fi

model_base_path=$1
output_base_dir=$2
train_microservice_id=$3
test_microservice_id=$4

should_reverse_data_input=${5:-"false"}
if [ "$should_reverse_data_input" = "true" ]; then
  should_reverse_data_flag="--reverse_test_data"
else
  should_reverse_data_flag=""
fi

model_path="${model_base_path}/workload_prediction_results_${train_microservice_id}"

if [ "$should_reverse_data_input" = "true" ]; then
  should_reverse_file_suffix="_reversed"
else
  should_reverse_file_suffix=""
fi

output_path="${output_base_dir}/microservice_train_${train_microservice_id}_test_${test_microservice_id}${should_reverse_file_suffix}"

python3 noisy_test.py --model_path "${model_path}/univariate_without_embedding/results/informer_sl12_ll6_pl3_dm1024_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue/model.pth" --output_dir "${output_path}/univariate_3_without_embedding" $should_reverse_data_flag --seq_len 12 --label_len 6 --pred_len 3 --test_microservice_id 25 --d_model 1024
python3 noisy_test.py --model_path "${model_path}/univariate_with_embedding/results/informer_sl12_ll6_pl3_dm1024_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue/model.pth" --output_dir "${output_path}/univariate_3_with_embedding" $should_reverse_data_flag --seq_len 12 --label_len 6 --pred_len 3 --test_microservice_id 25 --use_temporal_embedding --d_model 1024

python3 noisy_test.py --model_path "${model_path}/univariate_without_embedding_long_range_pred/results/informer_sl48_ll24_pl12_dm1024_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue/model.pth" --output_dir "${output_path}/univariate_12_without_embedding" $should_reverse_data_flag --seq_len 48 --label_len 24 --pred_len 12 --test_microservice_id 25 --d_model 1024
python3 noisy_test.py --model_path "${model_path}/univariate_with_embedding_long_range_pred/results/informer_sl48_ll24_pl12_dm1024_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue/model.pth" --output_dir "${output_path}/univariate_12_with_embedding" $should_reverse_data_flag --seq_len 48 --label_len 24 --pred_len 12 --test_microservice_id 25 --use_temporal_embedding --d_model 1024

python3 noisy_test.py --model_path "${model_path}/univariate_without_embedding_long_range_pred_24/results/informer_sl96_ll48_pl24_dm1024_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue/model.pth" --output_dir "${output_path}/univariate_24_without_embedding" $should_reverse_data_flag --seq_len 96 --label_len 48 --pred_len 24 --test_microservice_id 25 --d_model 1024
python3 noisy_test.py --model_path "${model_path}/univariate_with_embedding_long_range_pred_24/results/informer_sl96_ll48_pl24_dm1024_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue/model.pth" --output_dir "${output_path}/univariate_24_with_embedding" $should_reverse_data_flag --seq_len 96 --label_len 48 --pred_len 24 --test_microservice_id 25 --use_temporal_embedding --d_model 1024

python3 noisy_test.py --model_path "${model_path}/univariate_without_embedding_long_range_pred_60/results/informer_sl96_ll48_pl24_dm1024_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue/model.pth" --output_dir "${output_path}/univariate_60_without_embedding" $should_reverse_data_flag --seq_len 240 --label_len 120 --pred_len 60 --test_microservice_id 25 --d_model 1024
python3 noisy_test.py --model_path "${model_path}/univariate_with_embedding_long_range_pred_60/results/informer_sl96_ll48_pl24_dm1024_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue/model.pth" --output_dir "${output_path}/univariate_60_with_embedding" $should_reverse_data_flag --seq_len 240 --label_len 120 --pred_len 60 --test_microservice_id 25 --use_temporal_embedding --d_model 1024

python3 noisy_test.py --model_path "${model_path}/multivariate_without_embedding/results/informer_sl12_ll6_pl3_dm2048_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue/model.pth" --output_dir "${output_path}/multivariate_3_without_embedding" $should_reverse_data_flag --seq_len 12 --label_len 6 --pred_len 3 --d_model 2048
python3 noisy_test.py --model_path "${model_path}/multivariate_with_embedding/results/informer_sl12_ll6_pl3_dm2048_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue/model.pth" --output_dir "${output_path}/multivariate_3_with_embedding" $should_reverse_data_flag --seq_len 12 --label_len 6 --pred_len 3 --use_temporal_embedding --d_model 2048
