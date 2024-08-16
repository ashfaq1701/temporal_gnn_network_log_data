import math

import torch
from torch import nn


class WorkloadPredictionModel(torch.nn.Module):
    def __init__(
            self,
            n_past,
            n_future,
            embedding_dims,
            n_nodes,
            dropout=None,
            n_head=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=512,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.n_past = n_past
        self.n_future = n_future
        self.embedding_dims = embedding_dims
        self.n_nodes = n_nodes

        # Input embedding
        self.input_embedding = nn.Linear(n_nodes, n_past, embedding_dims)

        # Node Embedding
        self.node_embedding = nn.Embedding(n_nodes, n_nodes)

        d_model = embedding_dims + n_nodes

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout)

    def predict_workload(self, embeddings, nodes):
        pass

    def forward(self, x):
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
