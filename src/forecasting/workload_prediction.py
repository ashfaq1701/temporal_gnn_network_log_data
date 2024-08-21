import os
import time

import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from src.forecasting.informer.data.dataset import WorkloadPredictionDataset
from src.forecasting.informer.models.model import Informer, InformerStack
from src.forecasting.informer.utils.tools import EarlyStopping, adjust_learning_rate
from src.preprocess.functions import get_filtered_node_label_encoder, get_filtered_nodes_count
from src.utils import get_training_and_validation_file_indices


def predict_workload(args, ignore_temporal_embedding, result_path, only_use_target_microservice):
    target_microservice = os.getenv('WORKLOAD_PREDICTION_TARGET_MICROSERVICE')
    label_encoder = get_filtered_node_label_encoder()
    target_microservice_id = label_encoder.transform([target_microservice])[0]

    training_days = args.workload_pred_train_days
    validation_days = args.workload_pred_valid_days

    (train_start_minute, train_end_minute), (valid_start_minute, valid_end_minute) = \
        get_training_and_validation_file_indices(training_days, validation_days)

    node_count = get_filtered_nodes_count()
    d_embedding = args.memory_dim

    if torch.cuda.is_available():
        device_string = 'cuda:{}'.format(args.gpu)
    else:
        device_string = 'cpu'
    device = torch.device(device_string)

    setting = '{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}'.format(
        args.model, args.seq_len, args.label_len, args.pred_len, args.d_model, args.n_heads,
        args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, args.embed, args.distil, args.mix
    )

    workload_prediction = WorkloadTimeSeriesPrediction(
        args=args,
        train_start=train_start_minute,
        train_end=train_end_minute,
        valid_start=valid_start_minute,
        valid_end=valid_end_minute,
        node_count=node_count,
        d_embedding=d_embedding,
        use_temporal_embedding=not ignore_temporal_embedding,
        device=device,
        output_dir=result_path,
        target_node_id=target_microservice_id if only_use_target_microservice else None
    )

    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    workload_prediction.train(setting)

    print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    workload_prediction.predict(setting, True)


class WorkloadTimeSeriesPrediction:
    def __init__(
            self,
            args,
            train_start,
            train_end,
            valid_start,
            valid_end,
            node_count,
            d_embedding,
            use_temporal_embedding,
            device,
            output_dir,
            target_node_id=None
    ):
        self.args = args
        self.device = device
        self.output_dir = output_dir

        self.train_ds = WorkloadPredictionDataset(
            n_nodes=node_count,
            d_embed=d_embedding,
            is_train=True,
            train_start_minute=train_start,
            train_end_minute=train_end,
            valid_start_minute=valid_start,
            valid_end_minute=valid_end,
            seq_len=args.seq_len,
            label_len=args.label_len,
            pred_len=args.pred_len,
            use_temporal_embedding=use_temporal_embedding,
            node_id=target_node_id
        )

        self.valid_ds = WorkloadPredictionDataset(
            n_nodes=node_count,
            d_embed=d_embedding,
            is_train=False,
            train_start_minute=train_start,
            train_end_minute=train_end,
            valid_start_minute=valid_start,
            valid_end_minute=valid_end,
            seq_len=args.seq_len,
            label_len=args.label_len,
            pred_len=args.pred_len,
            use_temporal_embedding=use_temporal_embedding,
            node_id=target_node_id
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
                e_layers,  # self.args.e_layers,
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
        if flag == 'pred':
            shuffle_flag = False
            drop_last = False
            batch_size = 1
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = self.args.batch_size

        if flag == 'train':
            dataset = self.train_ds
        else:
            dataset = self.valid_ds

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=self.args.num_workers,
            drop_last=drop_last)

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
            pred, true = self._process_one_batch(vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.output_dir, self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

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
                pred, true = self._process_one_batch(train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
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

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss
            ))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.output_dir, self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()

        preds = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
            pred, true = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        print(preds.shape)

        # result save
        folder_path = os.path.join(self.output_dir, 'results/', setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + '/' + 'real_prediction.npy', preds)

        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        if self.args.padding == 0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding == 1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        else:
            raise ValueError("Invalid padding value.")

        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
        return outputs, batch_y
