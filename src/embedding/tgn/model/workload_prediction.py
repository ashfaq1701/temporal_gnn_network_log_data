import torch


class WorkloadPredictionModel(torch.nn.Module):
    def __init__(self, n_past, n_future, embedding_dims, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_past = n_past
        self.n_future = n_future
        self.embedding_dims = embedding_dims

    def forward(self, x):
        return x
