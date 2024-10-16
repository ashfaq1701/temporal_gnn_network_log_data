import os
import pickle
import time

import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from src.forecasting.informer.data.custom_max_scaler import CustomMaxScaler
from src.forecasting.informer.data.custom_std_scaler import CustomStandardScaler
from src.forecasting.informer.data.dataset import WorkloadPredictionDataset
from src.forecasting.informer.models.model import Informer, InformerStack
from src.forecasting.informer.utils.metrics import metric
from src.forecasting.informer.utils.tools import EarlyStopping, adjust_learning_rate
from src.preprocess.functions import get_filtered_nodes_count
from src.utils import get_target_microservice_id, get_training_validation_and_test_file_indices


def predict_workload(
        args, ignore_temporal_embedding, result_path, only_use_target_microservice
):
    if args.microservice_id is None:
        target_microservice_id = get_target_microservice_id()
    else:
        target_microservice_id = args.microservice_id

    if args.test_microservice_id is not None and args.test_microservice_id != 'none':
        test_microservice_id = int(args.test_microservice_id)
    else:
        test_microservice_id = target_microservice_id

    reversed_str = "_reversed" if args.should_reverse_data else ""
    output_base_path = f"{result_path}/workload_prediction_results_train_{target_microservice_id}_test_{test_microservice_id}{reversed_str}"

    output_folder_name = "univariate" if only_use_target_microservice else "multivariate_"
    output_folder_name = f"{output_folder_name}_{'without' if ignore_temporal_embedding else 'with'}_embedding"
    output_folder_name = f"{output_folder_name}_past_{args.seq_len}_future_{args.pred_len}"
    output_full_path = os.path.join(output_base_path, output_folder_name)

    training_days = args.workload_pred_train_days
    validation_days = args.workload_pred_valid_days
    test_days = args.workload_pred_test_days

    (train_start, train_end), (valid_start, valid_end), (test_start, test_end) = \
        get_training_validation_and_test_file_indices(
            training_days, validation_days, test_days, args.should_reverse_data
        )

    node_count = get_filtered_nodes_count()

    if torch.cuda.is_available():
        device_string = 'cuda:{}'.format(args.gpu)
    else:
        device_string = 'cpu'
    device = torch.device(device_string)

    setting = '{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}'.format(
        args.model, args.seq_len, args.label_len, args.pred_len, args.d_model, args.n_heads,
        args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, args.embed, args.distil, args.mix
    )

    results = []
    mses = []

    for run in range(args.n_runs):
        workload_prediction = WorkloadTimeSeriesPrediction(
            args=args,
            train_start=train_start,
            train_end=train_end,
            valid_start=valid_start,
            valid_end=valid_end,
            test_start=test_start,
            test_end=test_end,
            node_count=node_count,
            use_temporal_embedding=not ignore_temporal_embedding,
            device=device,
            output_dir=output_full_path,
            target_node_id=target_microservice_id if only_use_target_microservice else None,
            test_node_id=test_microservice_id if only_use_target_microservice else None
        )

        print('>>>>>>>start training {} : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(run, setting))
        model, losses_i = workload_prediction.train(setting)

        print('>>>>>>>start testing {} : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(run, setting))
        metrics_i, preds_i, trues_i = workload_prediction.test(setting)

        print('>>>>>>>predicting {} : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(run, setting))
        test_preds_inverted_i = workload_prediction.predict(setting, True)

        results.append((model, losses_i, metrics_i, preds_i, trues_i, test_preds_inverted_i))
        mses.append(metrics_i[1])

    min_idx = np.argmin(mses)
    model, losses, metrics, preds, trues, test_preds_inverted = results[min_idx]

    print(f'Best Run {min_idx}. MAE: {metrics[0]}, MSE: {metrics[1]}, RMSE: {metrics[2]}, MAPE: {metrics[3]}, MSPE: {metrics[4]}')

    folder_path = os.path.join(output_full_path, 'results/', setting)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    torch.save(model.state_dict(), os.path.join(folder_path, 'model.pth'))
    with open(os.path.join(folder_path, 'losses.pickle'), 'wb') as f:
        pickle.dump(losses, f)
    np.save(os.path.join(folder_path, 'metrics.npy'), metrics)
    np.save(os.path.join(folder_path, 'pred.npy'), preds)
    np.save(os.path.join(folder_path, 'true.npy'), trues)
    np.save(os.path.join(folder_path, 'real_prediction.npy'), test_preds_inverted)


class WorkloadTimeSeriesPrediction:
    def __init__(
            self,
            args,
            train_start,
            train_end,
            valid_start,
            valid_end,
            test_start,
            test_end,
            node_count,
            use_temporal_embedding,
            device,
            output_dir,
            target_node_id=None,
            test_node_id=None,
    ):
        self.args = args
        self.device = device
        self.output_dir = output_dir

        workload_scaler = build_workload_scaler(args.scale_workloads_per_feature)
        embedding_scaler = build_embedding_scaler(args.embedding_scaling_type, args.embedding_scaling_factor)

        train_workloads = get_scaled_workloads(train_start, train_end, workload_scaler, target_node_id, node_count)
        valid_workloads = get_scaled_workloads(valid_start, valid_end, workload_scaler, target_node_id, node_count)
        test_workloads = get_scaled_workloads(test_start, test_end, workload_scaler, test_node_id, node_count)

        train_embeddings = None
        valid_embeddings = None
        test_embeddings = None

        if use_temporal_embedding:
            train_embeddings = get_scaled_embeddings(train_start, train_end, embedding_scaler, target_node_id, node_count)
            valid_embeddings = get_scaled_embeddings(valid_start, valid_end, embedding_scaler, target_node_id, node_count)
            test_embeddings = get_scaled_embeddings(test_start, test_end, embedding_scaler, test_node_id, node_count)

        self.train_ds = WorkloadPredictionDataset(
            workloads=train_workloads,
            embeddings=train_embeddings,
            start_minute=train_start,
            end_minute=train_end,
            seq_len=args.seq_len,
            label_len=args.label_len,
            pred_len=args.pred_len,
            workload_scaler=workload_scaler
        )

        self.valid_ds = WorkloadPredictionDataset(
            workloads=valid_workloads,
            embeddings=valid_embeddings,
            start_minute=valid_start,
            end_minute=valid_end,
            seq_len=args.seq_len,
            label_len=args.label_len,
            pred_len=args.pred_len,
            workload_scaler=workload_scaler
        )

        self.test_ds = WorkloadPredictionDataset(
            workloads=test_workloads,
            embeddings=test_embeddings,
            start_minute=test_start,
            end_minute=test_end,
            seq_len=args.seq_len,
            label_len=args.label_len,
            pred_len=args.pred_len,
            workload_scaler=workload_scaler
        )

        n_features, n_labels = self.train_ds.get_feature_and_label_count()
        self.n_features = n_features
        self.n_labels = n_labels
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        model_dict = {
            'informer': Informer,
            'informerstack': InformerStack,
        }

        if self.args.model == 'informer' or self.args.model == 'informerstack':
            e_layers = self.args.e_layers if self.args.model == 'informer' else self.args.s_layers

            model = model_dict[self.args.model](
                self.n_features,
                self.n_labels,
                self.n_labels,
                self.args.seq_len,
                self.args.label_len,
                self.args.pred_len,
                self.args.factor,
                self.args.d_model,
                self.args.n_heads,
                e_layers,
                self.args.d_layers,
                self.args.d_ff,
                self.args.dropout,
                self.args.attn,
                self.args.embed,
                'm',
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()

            return model

    def _get_data(self, flag):
        if flag == 'test':
            shuffle_flag = False
            drop_last = True
            batch_size = self.args.batch_size
        elif flag == 'pred':
            shuffle_flag = False
            drop_last = False
            batch_size = 1
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = self.args.batch_size

        if flag == 'train':
            dataset = self.train_ds
        elif flag == 'val':
            dataset = self.valid_ds
        else:
            dataset = self.test_ds

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=self.args.num_workers,
            drop_last=drop_last
        )

        return dataset, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            pred, true = process_one_batch(
                batch_x, batch_y, batch_x_mark, batch_y_mark, self.model, self.args, self.device
            )
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.output_dir, self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        train_losses, valid_losses = [], []

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1

                model_optim.zero_grad()
                pred, true = process_one_batch(
                    batch_x, batch_y, batch_x_mark, batch_y_mark, self.model, self.args, self.device
                )
                loss = criterion(pred, true)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            train_losses.append(train_loss)
            valid_losses.append(vali_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss
            ))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))

        losses = {'train': train_losses, 'valid': valid_losses}

        return self.model, losses

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')

        self.model.eval()

        preds = []
        trues = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            pred, true = process_one_batch(
                batch_x, batch_y, batch_x_mark, batch_y_mark, self.model, self.args, self.device
            )
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        accuracy, mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('accuracy: {}, mse:{}, mae:{}'.format(accuracy, mse, mae))

        return np.array([accuracy, mae, mse, rmse, mape, mspe]), preds, trues

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.output_dir, self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()

        preds = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
            pred, true = process_one_batch(
                batch_x, batch_y, batch_x_mark, batch_y_mark, self.model, self.args, self.device
            )
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        preds_inverted = pred_data.inverse_transform(preds)
        return preds_inverted


def process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark, model, args, device):
    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float()

    batch_x_mark = batch_x_mark.float().to(device)
    batch_y_mark = batch_y_mark.float().to(device)

    # decoder input
    if args.padding == 0:
        dec_inp = torch.zeros([batch_y.shape[0], args.pred_len, batch_y.shape[-1]]).float()
    elif args.padding == 1:
        dec_inp = torch.ones([batch_y.shape[0], args.pred_len, batch_y.shape[-1]]).float()
    else:
        raise ValueError("Invalid padding value.")

    dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
    # encoder - decoder
    if args.use_amp:
        with torch.cuda.amp.autocast():
            if args.output_attention:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    else:
        if args.output_attention:
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        else:
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    batch_y = batch_y[:, -args.pred_len:, :].to(device)
    return outputs, batch_y


def build_workload_scaler(scale_workloads_per_feature):
    return CustomStandardScaler(scale_workloads_per_feature)


def build_embedding_scaler(embedding_scaling_type, embedding_scaling_factor):
    if embedding_scaling_type == 'max':
        embedding_scaler = CustomMaxScaler(scaling_factor=embedding_scaling_factor)
    elif embedding_scaling_type == 'std':
        embedding_scaler = CustomStandardScaler(per_feature=False, scaling_factor=embedding_scaling_factor)
    else:
        embedding_scaler = None

    return embedding_scaler


def get_scaled_workloads(start_minute, end_minute, workload_scaler, node_id, node_count):
    embedding_dir = os.getenv('EMBEDDING_DIR')

    with open(os.path.join(embedding_dir, 'workloads_over_time.pickle'), 'rb') as f:
        workloads = pickle.load(f)
        workloads = np.array(workloads)

    node_ids = [node_id] if node_id is not None else list(range(node_count))
    selected_workloads = workloads[:, node_ids]
    scaled_workloads = workload_scaler.fit_transform(selected_workloads)

    return scaled_workloads[start_minute:end_minute, :]


def get_scaled_embeddings(start_minute, end_minute, embedding_scaler, node_id, node_count):
    def scale_embeddings(embeddings):
        if embedding_scaler is None:
            return embeddings

        n_timesteps, n_nodes, n_features = embeddings.shape
        reshaped_data = embeddings.reshape((1, -1))
        scaled_data = embedding_scaler.fit_transform(reshaped_data)
        scaled_data_3d = scaled_data.reshape(n_timesteps, n_nodes, n_features)
        return scaled_data_3d

    embedding_dir = os.getenv('EMBEDDING_DIR')

    with open(os.path.join(embedding_dir, 'embeddings_over_time.pickle'), 'rb') as f:
        all_embeddings = pickle.load(f)
        all_embeddings = np.array(all_embeddings)

    node_ids = [node_id] if node_id is not None else list(range(node_count))
    selected_embeddings = all_embeddings[:, node_ids, :]
    scaled_embeddings = scale_embeddings(selected_embeddings)
    return scaled_embeddings[start_minute:end_minute, :, :]
