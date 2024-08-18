from collections import deque

import numpy as np


class EmbeddingBuffer:
    def __init__(self, n_nodes, embedding_dim, buffer_size):
        self.n_nodes = n_nodes
        self.embedding_dim = embedding_dim
        self.buffer_size = buffer_size
        self.embedding_store = np.zeros((self.n_nodes, self.embedding_dim), dtype=np.float32)
        self.node_embedding_buffer = None
        self.node_id_buffer = None
        self.past_workload_buffer = None
        self.future_workload_buffer = None
        self.timestep_buffer = None

    def add_embeddings(self, nodes, embeddings):
        for idx in range(len(nodes)):
            self.embedding_store[nodes[idx], :] = embeddings[idx, :]

    def add_to_buffer(self, current_timestep, past_workloads, future_workloads):
        shuffled_indices = np.random.permutation(self.n_nodes)
        shuffled_embeddings = self.embedding_store[shuffled_indices, :]
        shuffled_past_workloads = past_workloads[shuffled_indices, :]
        shuffled_future_workloads = future_workloads[shuffled_indices, :]
        current_timestep_values = np.full(self.n_nodes, current_timestep)

        if self.node_embedding_buffer is not None:
            self.node_embedding_buffer = np.concatenate((self.node_embedding_buffer, shuffled_embeddings))
        else:
            self.node_embedding_buffer = shuffled_embeddings

        if self.past_workload_buffer is not None:
            self.past_workload_buffer = np.concatenate((self.past_workload_buffer, shuffled_past_workloads))
        else:
            self.past_workload_buffer = shuffled_past_workloads

        if self.future_workload_buffer is not None:
            self.future_workload_buffer = np.concatenate((self.future_workload_buffer, shuffled_future_workloads))
        else:
            self.future_workload_buffer = shuffled_future_workloads

        if self.node_id_buffer is not None:
            self.node_id_buffer = np.concatenate((self.node_id_buffer, shuffled_indices))
        else:
            self.node_id_buffer = shuffled_indices

        if self.timestep_buffer is not None:
            self.timestep_buffer = np.concatenate((self.timestep_buffer, current_timestep_values))
        else:
            self.timestep_buffer = current_timestep_values

    def is_buffer_full(self):
        return self.node_id_buffer.shape[0] >= self.buffer_size

    def get_batch(self):
        if self.node_embedding_buffer is None:
            return np.array([]), np.array([])

        embedding_batch = self.node_embedding_buffer[:self.buffer_size, :]
        nodes_batch = self.node_id_buffer[:self.buffer_size]
        past_workload_batch = self.past_workload_buffer[:self.buffer_size]
        future_workload_batch = self.future_workload_buffer[:self.buffer_size]
        timestep_batch = self.timestep_buffer[:self.buffer_size]

        self.node_embedding_buffer = self.node_embedding_buffer[self.buffer_size:, :]
        self.node_id_buffer = self.node_id_buffer[self.buffer_size:]
        self.past_workload_buffer = self.past_workload_buffer[self.buffer_size:]
        self.future_workload_buffer = self.future_workload_buffer[self.buffer_size:]
        self.timestep_buffer = self.timestep_buffer[self.buffer_size:]

        return embedding_batch, nodes_batch, past_workload_batch, future_workload_batch, timestep_batch

    def reset_store(self):
        self.embedding_store = np.zeros((self.n_nodes, self.embedding_dim), dtype=np.float32)
        self.node_embedding_buffer = None
        self.node_id_buffer = None
        self.past_workload_buffer = None
        self.future_workload_buffer = None
        self.timestep_buffer = None
