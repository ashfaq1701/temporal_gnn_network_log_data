import logging
import math
import os.path
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from src.embedding.tgn.evaluation.evaluation import eval_edge_prediction
from src.embedding.tgn.model.tgn import TGN
from src.embedding.tgn.utils.data_processing import get_data
from src.embedding.tgn.utils.utils import RandEdgeSampler, EarlyStopMonitor, get_neighbor_finder
from src.preprocess.compute_time_statistics import compute_time_shifts_for_n_days


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

    training_days =  int(os.getenv('TRAINING_DAYS'))
    validation_days = int(os.getenv('VALIDATION_DAYS'))

    ### Extract data for training, validation and testing
    node_features, edge_features, full_data, train_data, val_data = get_data(training_days, validation_days)

    # Initialize training neighbor finder to retrieve temporal graph
    train_ngh_finder = get_neighbor_finder(train_data, args.uniform)

    # Initialize validation and test neighbor finder to retrieve temporal graph
    full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

    # Initialize negative samplers. Set seeds for validation and testing so negatives are the same
    # across different runs
    # NB: in the inductive setting, negatives are sampled only amongst other new nodes
    train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
    val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)

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

    # Set device
    device_string = 'cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_string)

    # Compute time statistics
    mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
        compute_time_shifts_for_n_days(14)

    for i in range(args.n_runs):
        results_path = "results/{}_{}.pkl".format(args.prefix, i) if i > 0 else "results/{}.pkl".format(args.prefix)
        Path("results/").mkdir(parents=True, exist_ok=True)

        # Initialize Model
        tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
                  edge_features=edge_features, device=device,
                  n_layers=num_layer,
                  n_heads=num_heads, dropout=drop_out, use_memory=use_memory,
                  message_dimension=message_dim, memory_dimension=memory_dim,
                  memory_update_at_start=not args.memory_update_at_end,
                  embedding_module_type=args.embedding_module,
                  message_function=args.message_function,
                  aggregator_type=args.aggregator,
                  memory_updater_type=args.memory_updater,
                  n_neighbors=num_neighbors,
                  mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
                  mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
                  use_destination_embedding_in_message=args.use_destination_embedding_in_message,
                  use_source_embedding_in_message=args.use_source_embedding_in_message,
                  dyrep=args.dyrep)
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(tgn.parameters(), lr=learning_rate)
        tgn = tgn.to(device)

        num_instance = len(train_data.sources)
        num_batch = math.ceil(num_instance / batch_size)

        logger.info('num of training instances: {}'.format(num_instance))
        logger.info('num of batches per epoch: {}'.format(num_batch))
        idx_list = np.arange(num_instance)

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

            # Train using only training graph
            tgn.set_neighbor_finder(train_ngh_finder)
            m_loss = []

            logger.info('start {} epoch'.format(epoch))
            for k in range(0, num_batch, args.backprop_every):
                loss = 0
                optimizer.zero_grad()

                # Custom loop to allow to perform backpropagation only every a certain number of batches
                for j in range(args.backprop_every):
                    batch_idx = k + j

                    if batch_idx >= num_batch:
                        continue

                    start_idx = batch_idx * batch_size
                    end_idx = min(num_instance, start_idx + batch_size)
                    sources_batch, destinations_batch = train_data.sources[start_idx:end_idx], \
                        train_data.destinations[start_idx:end_idx]
                    edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
                    timestamps_batch = train_data.timestamps[start_idx:end_idx]

                    size = len(sources_batch)
                    _, negatives_batch = train_rand_sampler.sample(size)

                    with torch.no_grad():
                        pos_label = torch.ones(size, dtype=torch.float, device=device)
                        neg_label = torch.zeros(size, dtype=torch.float, device=device)

                    tgn = tgn.train()
                    pos_prob, neg_prob = tgn.compute_edge_probabilities(sources_batch, destinations_batch,
                                                                        negatives_batch,
                                                                        timestamps_batch, edge_idxs_batch,
                                                                        num_neighbors)

                    loss += criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)

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

            ### Validation
            # Validation uses the full graph
            tgn.set_neighbor_finder(full_ngh_finder)

            if use_memory:
                # Backup memory at the end of training, so later we can restore it and use it for the
                # validation on unseen nodes
                train_memory_backup = tgn.memory.backup_memory()

            val_ap, val_auc = eval_edge_prediction(model=tgn,
                                                   negative_edge_sampler=val_rand_sampler,
                                                   data=val_data,
                                                   n_neighbors=num_neighbors)
            if use_memory:
                val_memory_backup = tgn.memory.backup_memory()
                # Restore memory we had at the end of training to be used when validating on new nodes.
                # Also backup memory after validation so it can be used for testing (since test edges are
                # strictly later in time than validation edges)
                tgn.memory.restore_memory(train_memory_backup)

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
        if use_memory:
            # Restore memory at the end of validation (save a model which is ready for testing)
            tgn.memory.restore_memory(val_memory_backup)
        torch.save(tgn.state_dict(), model_save_path)
        logger.info('TGN model saved')
