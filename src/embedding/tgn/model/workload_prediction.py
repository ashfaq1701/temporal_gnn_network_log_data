import math

import torch
import torch.nn.functional as F
from torch import nn


class WorkloadPredictionModel(nn.Module):
    def __init__(
            self, m, n, n_nodes, d_embed, device, d_node_embed=128, nhead=4,
            num_encoder_layers=3, dim_feedforward=2048, dropout=0.1
    ):
        super(WorkloadPredictionModel, self).__init__()

        self.m = m
        self.n = n
        self.n_nodes = n_nodes
        self.device = device

        self.node_embedding = nn.Linear(n_nodes, d_node_embed)  # Project one-hot node vectors to embedding dimension
        d_model = d_node_embed + d_embed

        self.pos_encoder = PositionalEncoding(d_model, max_len=m)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)

        self.fc_out = nn.Linear(d_model, 1)

    def predict_workload(self, embeddings, nodes):
        embeddings_torch = torch.from_numpy(embeddings).to(self.device)
        nodes_torch = torch.from_numpy(nodes).to(self.device)
        nodes_one_hot = F.one_hot(nodes_torch, num_classes=self.n_nodes).float()
        return self(embeddings_torch, nodes_one_hot)

    def forward(self, x_past, node_ids):
        b, m, d = x_past.shape

        node_embeddings = self.node_embedding(node_ids)  # Shape: (b, d_model)
        node_embeddings = node_embeddings.unsqueeze(1).expand(-1, m, -1)  # Shape: (b, m, d_model)

        x = torch.cat((x_past, node_embeddings), dim=2)  # Shape: (b, m, d_model)

        x = self.pos_encoder(x)  # Shape: (b, m, d_model)

        # Pass through transformer encoder
        x = x.permute(1, 0, 2)  # Change to (seq_len, batch_size, d_model)
        transformer_output = self.transformer_encoder(x)  # Shape: (seq_len, batch_size, d_model + embedding_dim)
        transformer_output = transformer_output.permute(1, 0, 2)  # Change back to (batch_size, seq_len, d_model)

        y_pred = self.fc_out(transformer_output)  # Shape: (b, m, output_dim)

        y_pred = y_pred[:, -self.n:, :]  # Shape: (b, n, output_dim)

        return y_pred.squeeze()


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
