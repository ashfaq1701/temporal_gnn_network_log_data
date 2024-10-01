import argparse

from dotenv import load_dotenv

from src.forecasting.workload_pred_noisy_data import test_workload_pred_noisy_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alibaba Graph Training.")

    parser.add_argument('--model_path', type=str, required=True, help='The model path.')
    parser.add_argument('--test_microservice_id', type=int, required=True, help='Test microservice id.')
    parser.add_argument('--test_days', type=int, default=3, help='Test days in workload prediction')
    parser.add_argument('--total_days', type=int, default=14, help='Total days in workload prediction')
    parser.add_argument('--reverse_test_data', action='store_true', help='Should we reverse the test data.')
    parser.add_argument('--num_datasets', type=int, default=11, help='Number of noisy runs.')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42], help='List of seeds for the noise generator (default: [42]).')

    parser.add_argument('--seq_len', type=int, default=12, help='Number of past timesteps to predict from.')
    parser.add_argument('--label_len', type=int, default=6, help='Number of future timesteps to predict')
    parser.add_argument('--pred_len', type=int, default=3, help='Number of future timesteps to predict')

    parser.add_argument('--embedding_scaling_type', type=str, default='max', help='Type of embedding scaling, either none, std or max')
    parser.add_argument('--embedding_scaling_factor', type=float, default=None, help='Factor to scale down the temporal embedding')

    parser.add_argument('--use_temporal_embedding', action='store_true', help='Should we use temporal embedding?')

    parser.add_argument('--scale_workloads_per_feature', action='store_true', help='If we need to scale workload per feature (default is globally)')

    args = parser.parse_args()

    load_dotenv()

    test_workload_pred_noisy_data(
        args.model_path, args.seq_len, args.label_len, args.pred_len, args.test_microservice_id, args.test_days,
        args.use_temporal_embedding, args.reverse_test_data, args.num_datasets, args.seeds,
        args.scale_workloads_per_feature, args.embedding_scaling_type, args.embedding_scaling_factor, args.total_days
    )
