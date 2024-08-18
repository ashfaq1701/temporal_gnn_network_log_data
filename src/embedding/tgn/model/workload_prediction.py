import torch
import torch.nn as nn
import torch.nn.functional as F


class WorkloadPrediction(nn.Module):
    def __init__(self, n_nodes, device):
        super(WorkloadPrediction, self).__init__()
        self.n_nodes = n_nodes
        self.device = device

    def predict_workload(self, embeddings, nodes):
        embeddings_torch = torch.from_numpy(embeddings).to(self.device)
        nodes_torch = torch.from_numpy(nodes).to(self.device)
        nodes_one_hot = F.one_hot(nodes_torch, num_classes=self.n_nodes).float()
        return self(embeddings_torch, nodes_one_hot)


class WorkloadPredictionMLP(WorkloadPrediction):
    def __init__(self, n_future, d_embedding, n_nodes, device, d_node_embedding=128, hidden_dim=256, dropout_prob=0.0):
        super(WorkloadPredictionMLP, self).__init__(n_nodes, device)

        self.node_embedding = nn.Linear(n_nodes, d_node_embedding)

        layers = [
            nn.Linear(d_embedding + d_node_embedding, hidden_dim),
            nn.ReLU(),
        ]

        if dropout_prob > 0.0:
            layers.append(nn.Dropout(dropout_prob))

        layers.extend([
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        ])

        if dropout_prob > 0.0:
            layers.append(nn.Dropout(dropout_prob))

        layers.append(nn.Linear(hidden_dim, n_future))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x, nodes):
        node_emb = self.node_embedding(nodes)
        x = torch.cat([x, node_emb], dim=1)
        output = self.mlp(x)
        return output
