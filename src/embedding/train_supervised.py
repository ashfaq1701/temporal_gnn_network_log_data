import logging
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

from src.embedding.tgn.evaluation.evaluation import eval_workload_prediction
from src.embedding.tgn.model.tgn import TGN
from src.embedding.tgn.model.workload_prediction import WorkloadPredictionModel
from src.embedding.tgn.utils.data_processing import CombinedPandasDatasetFromDirectory
from src.embedding.tgn.utils.embedding_buffer import EmbeddingBuffer
from src.embedding.tgn.utils.utils import EarlyStopMonitor, get_unique_latest_nodes_with_indices, get_future_workloads
from src.embedding.train_self_supervised import get_upstream_and_downstream_nodes
from src.preprocess.compute_time_statistics import compute_time_shifts_for_n_days
from src.preprocess.functions import get_edge_feature_count, get_filtered_workload_counts
from src.utils import get_training_and_validation_file_indices

sys.path.append(os.path.abspath('./build'))
import neighbor_finder


def train_workload_prediction_model(args):
    batch_size = args.bs
    num_neighbors = args.n_degree
    num_neg = 1
    num_epoch = args.n_epoch
    num_heads = args.n_head
    drop_out = args.drop_out
    gpu = args.gpu
    uniform = args.uniform
    seq_len = num_neighbors
    num_layer = args.n_layer
    learning_rate = args.lr
    node_dim = args.node_dim
    time_dim = args.time_dim
    use_memory = args.use_memory
    message_dim = args.message_dim
    memory_dim = args.memory_dim
    n_past = args.n_past
    n_future = args.n_future

    results_dir = os.getenv('RESULTS_DIR')

    Path(os.path.join(results_dir, "saved_models")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(results_dir, "saved_checkpoints")).mkdir(parents=True, exist_ok=True)
    model_save_path = os.path.join(results_dir, f'saved_models/{args.prefix}-alibaba-node-classification.pth')
    get_checkpoint_path = lambda \
            epoch: f'saved_checkpoints/{args.prefix}-alibaba-{epoch}-node-classification.pth'

    ### set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(results_dir, 'log/{}.log').format(str(time.time())))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(args)

    data_directory = os.getenv('FILTERED_DATA_DIR')

    training_days = float(os.getenv('TRAIN_DAYS'))
    validation_days = float(os.getenv('VALID_DAYS'))

    (train_file_start_idx, train_file_end_idx), (valid_file_start_idx, valid_file_end_idx) = \
        get_training_and_validation_file_indices(training_days, validation_days)

    neighbor_buffer_duration_hours = int(os.getenv('NEIGHBOR_BUFFER_DURATION_HOURS'))
    n_edge_features = get_edge_feature_count()
    ngh_finder = neighbor_finder.NeighborFinder(neighbor_buffer_duration_hours, n_edge_features, args.uniform)

    train_dataset = CombinedPandasDatasetFromDirectory(
        data_directory,
        train_file_start_idx,
        train_file_end_idx,
        batch_size,
        ngh_finder,
        logger
    )

    valid_dataset = CombinedPandasDatasetFromDirectory(
        data_directory,
        valid_file_start_idx,
        valid_file_end_idx,
        batch_size,
        ngh_finder,
        logger
    )

    _, _, _, _, n_nodes = \
        get_upstream_and_downstream_nodes(
            train_file_start_idx,
            train_file_end_idx,
            valid_file_start_idx,
            valid_file_end_idx
        )

    workloads = get_filtered_workload_counts()
    embedding_buffer = EmbeddingBuffer(n_past, n_nodes, memory_dim)

    # Set device
    if torch.cuda.is_available():
        device_string = 'cuda:{}'.format(gpu)
    else:
        device_string = 'cpu'

        num_cores = os.cpu_count()
        print(f'Num CPU cores: {num_cores}')
        torch.set_num_threads(num_cores)
        torch.set_num_interop_threads(num_cores)

    device = torch.device(device_string)
    logger.info('Device: {}'.format(device))

    # Compute time statistics
    mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = compute_time_shifts_for_n_days(
        training_days + validation_days
    )

    for i in range(args.n_runs):
        results_path = os.path.join(
            results_dir,
            "results/{}_alibaba_node_classification_{}.pkl".format(
                args.prefix,
                i
            ) if i > 0 else "results/{}_alibaba_node_classification.pkl".format(args.prefix)
        )
        Path(os.path.join(results_dir, "results/")).mkdir(parents=True, exist_ok=True)

        # Initialize Model
        tgn = TGN(neighbor_finder=ngh_finder, n_node_features=memory_dim, n_nodes=n_nodes,
                  n_edge_features=n_edge_features, device=device, n_layers=num_layer, n_heads=num_heads,
                  dropout=drop_out, use_memory=use_memory, message_dimension=message_dim, memory_dimension=memory_dim,
                  memory_update_at_start=not args.memory_update_at_end, embedding_module_type=args.embedding_module,
                  message_function=args.message_function, aggregator_type=args.aggregator, n_neighbors=num_neighbors,
                  mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
                  mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
                  use_destination_embedding_in_message=args.use_destination_embedding_in_message,
                  use_source_embedding_in_message=args.use_source_embedding_in_message)

        tgn = tgn.to(device)

        logger.info('Loading saved TGN model')
        model_path = os.path.join(results_dir, f'saved_models/{args.prefix}-alibaba.pth')
        tgn.load_state_dict(torch.load(model_path, map_location=device))
        tgn.eval()
        logger.info('TGN models loaded')
        logger.info('Start training node classification task')

        workload_predictor = WorkloadPredictionModel(n_past, n_future, memory_dim)
        workload_predictor_optimizer = torch.optim.Adam(workload_predictor.parameters(), lr=args.lr)
        workload_predictor = workload_predictor.to(device)
        workload_predictor_loss_criterion = torch.nn.L1Loss()

        val_aucs = []
        epoch_times = []
        train_losses = []
        early_stopper = EarlyStopMonitor(max_round=args.patience)

        for epoch in range(args.n_epoch):
            start_epoch = time.time()

            # Initialize memory of the model at each epoch
            if use_memory:
                tgn.memory.__init_memory__()

            train_dataset.reset()
            valid_dataset.reset()
            ngh_finder.reset()
            embedding_buffer.reset_store()

            tgn = tgn.eval()
            loss = 0

            current_minute = 0

            for batch in train_dataset:
                sources_batch, destinations_batch, timestamps_batch, edge_idxs_batch, \
                    edge_features_batch, current_file_end = batch
                size = len(sources_batch)

                with torch.no_grad():
                    _, destination_embeddings, _ = tgn.compute_temporal_embeddings(sources_batch,
                                                                                   destinations_batch,
                                                                                   destinations_batch,
                                                                                   timestamps_batch,
                                                                                   edge_idxs_batch,
                                                                                   num_neighbors)
                    destination_embeddings_np = destination_embeddings.detach().cpu().numpy()

                train_dataset.add_batch_to_neighbor_finder(batch[:-1])

                nodes_with_latest_indices = get_unique_latest_nodes_with_indices(destinations_batch, timestamps_batch)
                latest_destination_embeddings = destination_embeddings_np[nodes_with_latest_indices[:, 1]]
                embedding_buffer.add_embeddings(nodes_with_latest_indices[:, 0], latest_destination_embeddings)

                if current_file_end:
                    embedding_buffer.flush_embeddings_to_store()

                    current_minute_workloads = get_future_workloads(
                        np.arange(0, n_nodes),
                        current_minute,
                        workloads,
                        n_future
                    )

                    # TODO: TRAIN THE MODEL
                    # pred = workload_predictor(source_embedding).sigmoid()
                    # decoder_loss = decoder_loss_criterion(pred, labels_batch_torch)
                    # decoder_loss.backward()
                    # decoder_optimizer.step()
                    # loss += decoder_loss.item()

                    current_minute += 1

            train_losses.append(loss / train_dataset.get_total_batches())

            val_auc = eval_workload_prediction(
                tgn=tgn,
                workload_predictor=workload_predictor,
                valid_dataset=valid_dataset,
                n_neighbors=num_neighbors
            )
            val_aucs.append(val_auc)

            epoch_time = time.time() - start_epoch
            epoch_times.append(epoch_time)

            pickle.dump({
                "val_aps": val_aucs,
                "train_losses": train_losses,
                "epoch_times": epoch_times
            }, open(results_path, "wb"))

            logger.info(
                f'Epoch {epoch}: train loss: {loss / train_dataset.get_total_batches()}, val auc: {val_auc}, time: {time.time() - start_epoch}')

        if args.use_validation:
            if early_stopper.early_stop_check(val_auc):
                logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
                break
            else:
                torch.save(workload_predictor.state_dict(), get_checkpoint_path(epoch))

        if args.use_validation:
            logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
            best_model_path = get_checkpoint_path(early_stopper.best_epoch)
            workload_predictor.load_state_dict(torch.load(best_model_path))
            logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
            workload_predictor.eval()

        pickle.dump({
            "val_aps": val_aucs,
            "train_losses": train_losses,
            "epoch_times": [0.0]
        }, open(results_path, "wb"))
