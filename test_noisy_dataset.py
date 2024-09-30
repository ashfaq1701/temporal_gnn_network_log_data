import argparse

from src.forecasting.workload_pred_noisy_data import test_workload_pred_noisy_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alibaba Graph Training.")

    parser.add_argument('--model_path', type=str, required=True, help='The model path.')
    parser.add_argument('--test_microservice_id', type=int, required=True, help='Test microservice id.')
    parser.add_argument('--reverse_test_data', action='store_true', help='Should we reverse the test data')
    parser.add_argument('--num_runs', type=int, default=10, help='Number of noisy runs.')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42], help='List of seeds for the noise generator (default: [42]).')

    args = parser.parse_args()

    test_workload_pred_noisy_data(
        args.model_path, args.test_microservice_id, args.reverse_test_data, args.num_runs, args.seeds
    )
