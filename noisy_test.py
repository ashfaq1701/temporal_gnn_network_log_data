import argparse

from dotenv import load_dotenv

from src.forecasting.workload_pred_noisy_data import test_workload_pred_noisy_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alibaba Graph Training.")

    parser.add_argument('--model_path', type=str, required=True, help='The model path.')
    parser.add_argument('--test_microservice_id', type=int, help='Test microservice id.')
    parser.add_argument('--output_dir', type=str, help='Output directory path')
    parser.add_argument('--test_days', type=int, default=3, help='Test days in workload prediction')
    parser.add_argument('--total_days', type=int, default=14, help='Total days in workload prediction')
    parser.add_argument('--reverse_test_data', action='store_true', help='Should we reverse the test data.')
    parser.add_argument('--num_noisy_iters', type=int, default=11, help='Number of noisy runs.')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42], help='List of seeds for the noise generator (default: [42]).')

    parser.add_argument('--seq_len', type=int, default=12, help='Number of past timesteps to predict from.')
    parser.add_argument('--label_len', type=int, default=6, help='Number of future timesteps to predict')
    parser.add_argument('--pred_len', type=int, default=3, help='Number of future timesteps to predict')

    parser.add_argument('--embedding_scaling_type', type=str, default='max', help='Type of embedding scaling, either none, std or max')
    parser.add_argument('--embedding_scaling_factor', type=float, default=None, help='Factor to scale down the temporal embedding')

    parser.add_argument('--use_temporal_embedding', action='store_true', help='Should we use temporal embedding?')

    parser.add_argument('--scale_workloads_per_feature', action='store_true', help='If we need to scale workload per feature (default is globally)')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
    parser.add_argument('--padding', type=int, default=0, help='padding type')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
    parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    args = parser.parse_args()

    load_dotenv()

    test_workload_pred_noisy_data(args)
