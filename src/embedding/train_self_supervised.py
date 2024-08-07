import logging
import os.path
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from src.embedding.tgn.evaluation.evaluation import eval_edge_prediction
from src.embedding.tgn.model.tgn import TGN
from src.embedding.tgn.utils.data_processing import CombinedPandasDatasetFromDirectory
from src.embedding.tgn.utils.utils import NeighborFinder, RandEdgeSampler, EarlyStopMonitor
from src.preprocess.compute_time_statistics import compute_time_shifts_for_n_days
from src.preprocess.functions import get_edge_feature_count, get_upstream_counts_object, get_downstream_counts_object, \
    get_node_label_encoder, get_encoded_nodes, get_filtered_node_label_encoder, get_filtered_stats
from src.utils import get_training_and_validation_file_indices


def train_link_prediction_model(args):
    batch_size = args.bs
    num_neighbors = args.n_degree
    num_neg = 1
    num_epoch = args.n_epoch
    num_heads = args.n_head
    drop_out = args.drop_out
    gpu = args.gpu
    num_layer = args.n_layer
    learning_rate = args.lr
    node_dim = args.node_dim
    time_dim = args.time_dim
    use_memory = args.use_memory
    message_dim = args.message_dim
    memory_dim = args.memory_dim

    Path("saved_models/").mkdir(parents=True, exist_ok=True)
    Path("saved_checkpoints/").mkdir(parents=True, exist_ok=True)
    model_save_path = f'saved_models/{args.prefix}.pth'
    get_checkpoint_path = lambda epoch: f'saved_checkpoints/{args.prefix}-{args.data}-{epoch}.pth'

    ### set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    Path("log/").mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
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

    training_days = int(os.getenv('TRAIN_DAYS'))
    validation_days = int(os.getenv('VALID_DAYS'))
    neighbor_buffer_duration_hours = int(os.getenv('NEIGHBOR_BUFFER_DURATION_HOURS'))

    (train_file_start_idx, train_file_end_idx), (valid_file_start_idx, valid_file_end_idx) = \
        get_training_and_validation_file_indices(training_days, validation_days)

    n_edge_features = get_edge_feature_count()
    n_node_features = int(os.getenv('N_NODE_FEATURES'))
    neighbor_finder = NeighborFinder(neighbor_buffer_duration_hours * 60 * 60, n_edge_features, args.uniform)

    train_dataset = CombinedPandasDatasetFromDirectory(
        data_directory,
        train_file_start_idx,
        train_file_end_idx,
        batch_size,
        neighbor_finder,
        logger
    )

    valid_dataset = CombinedPandasDatasetFromDirectory(
        data_directory,
        valid_file_start_idx,
        valid_file_end_idx,
        batch_size,
        neighbor_finder,
        logger
    )

    upstream_nodes_train, downstream_nodes_train, upstream_nodes_valid, downstream_nodes_valid, n_nodes = \
        get_upstream_and_downstream_nodes(
            train_file_start_idx,
            train_file_end_idx,
            valid_file_start_idx,
            valid_file_end_idx
        )

    train_rand_sampler = RandEdgeSampler(upstream_nodes_train, downstream_nodes_train)
    val_rand_sampler = RandEdgeSampler(upstream_nodes_valid, downstream_nodes_valid, seed=0)

    # Set device
    if torch.cuda.is_available():
        device_string = 'cuda:{}'.format(gpu)
    elif torch.backends.mps.is_available():
        device_string = 'mps'
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
        results_path = "results/{}_{}.pkl".format(args.prefix, i) if i > 0 else "results/{}.pkl".format(args.prefix)
        Path("results/").mkdir(parents=True, exist_ok=True)

        # Initialize Model
        tgn = TGN(n_node_features=n_node_features, n_nodes=n_nodes, n_edge_features=n_edge_features,
                  neighbor_finder=neighbor_finder, device=device, n_layers=num_layer, n_heads=num_heads,
                  dropout=drop_out, use_memory=use_memory, message_dimension=message_dim, memory_dimension=memory_dim,
                  memory_update_at_start=not args.memory_update_at_end, embedding_module_type=args.embedding_module,
                  message_function=args.message_function, aggregator_type=args.aggregator,
                  memory_updater_type=args.memory_updater, n_neighbors=num_neighbors,
                  mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
                  mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
                  use_destination_embedding_in_message=args.use_destination_embedding_in_message,
                  use_source_embedding_in_message=args.use_source_embedding_in_message,
                  dyrep=args.dyrep)
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(tgn.parameters(), lr=learning_rate)
        tgn = tgn.to(device)

        val_aps = []
        epoch_times = []
        total_epoch_times = []
        train_losses = []

        early_stopper = EarlyStopMonitor(max_round=args.patience)
        for epoch in range(num_epoch):
            start_epoch = time.time()
            ### Training

            # Reinitialize memory of the model at the start of each epoch
            if use_memory:
                tgn.memory.__init_memory__()

            neighbor_finder.reset()
            m_loss = []

            logger.info('start {} epoch'.format(epoch))

            backprop_running_count = 0
            loss = 0

            for batch in train_dataset:
                if backprop_running_count == 0:
                    loss = 0
                    optimizer.zero_grad()

                sources_batch, destinations_batch, timestamps_batch, edge_idxs_batch, edge_features_batch = batch
                size = len(sources_batch)
                _, negatives_batch = train_rand_sampler.sample(size)

                with torch.no_grad():
                    pos_label = torch.ones(size, dtype=torch.float, device=device)
                    neg_label = torch.zeros(size, dtype=torch.float, device=device)

                tgn = tgn.train()
                pos_prob, neg_prob = tgn.compute_edge_probabilities(
                    sources_batch,
                    destinations_batch,
                    negatives_batch,
                    timestamps_batch,
                    edge_features_batch,
                    num_neighbors
                )

                loss += criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)

                backprop_running_count += 1
                if backprop_running_count == args.backprop_every:
                    print(f"Backpropagating {backprop_running_count}")

                    backprop_running_count = 0

                    loss /= args.backprop_every

                    loss.backward()
                    optimizer.step()
                    m_loss.append(loss.item())

                    # Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to
                    # the start of time
                    if use_memory:
                        tgn.memory.detach_memory()

            if backprop_running_count > 0:
                loss /= args.backprop_every

                loss.backward()
                optimizer.step()
                m_loss.append(loss.item())

                # Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to
                # the start of time
                if use_memory:
                    tgn.memory.detach_memory()

            epoch_time = time.time() - start_epoch
            epoch_times.append(epoch_time)

            val_ap, val_auc = eval_edge_prediction(model=tgn,
                                                   negative_edge_sampler=val_rand_sampler,
                                                   valid_dataset=valid_dataset,
                                                   n_neighbors=num_neighbors)

            val_aps.append(val_ap)
            train_losses.append(np.mean(m_loss))

            # Save temporary results to disk
            pickle.dump({
                "val_aps": val_aps,
                "train_losses": train_losses,
                "epoch_times": epoch_times,
                "total_epoch_times": total_epoch_times
            }, open(results_path, "wb"))

            total_epoch_time = time.time() - start_epoch
            total_epoch_times.append(total_epoch_time)

            logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
            logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
            logger.info(
                'val auc: {}'.format(val_auc))
            logger.info(
                'val ap: {}'.format(val_ap))

            # Early stopping
            if early_stopper.early_stop_check(val_ap):
                logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
                logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
                best_model_path = get_checkpoint_path(early_stopper.best_epoch)
                tgn.load_state_dict(torch.load(best_model_path))
                logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
                tgn.eval()
                break
            else:
                torch.save(tgn.state_dict(), get_checkpoint_path(epoch))

        # Save results for this run
        pickle.dump({
            "val_aps": val_aps,
            "epoch_times": epoch_times,
            "train_losses": train_losses,
            "total_epoch_times": total_epoch_times
        }, open(results_path, "wb"))

        logger.info('Saving TGN model')
        torch.save(tgn.state_dict(), model_save_path)
        logger.info('TGN model saved')


def get_upstream_and_downstream_nodes(train_start_idx, train_end_idx, valid_start_idx, valid_end_idx):
    upstream_counts, downstream_counts, _ = get_filtered_stats()
    filtered_label_encoder = get_filtered_node_label_encoder()
    n_nodes = len(filtered_label_encoder.classes_)

    upstream_nodes_train, downstream_nodes_train = get_encoded_nodes(
        upstream_counts,
        downstream_counts,
        train_start_idx,
        train_end_idx
    )
    upstream_nodes_valid, downstream_nodes_valid = get_encoded_nodes(
        upstream_counts,
        downstream_counts,
        valid_start_idx,
        valid_end_idx
    )

    return upstream_nodes_train, downstream_nodes_train, upstream_nodes_valid, downstream_nodes_valid, n_nodes
