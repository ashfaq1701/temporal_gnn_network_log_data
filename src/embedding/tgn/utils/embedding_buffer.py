from collections import deque

import numpy as np


class EmbeddingBuffer:
    def __init__(self, n_past, n_nodes, embedding_dim, buffer_size):
        self.n_past = n_past
        self.n_nodes = n_nodes
        self.embedding_dim = embedding_dim
        self.buffer_size = buffer_size
        self.embedding_store = []
        self.current_embedding_store = {}
        self.node_embedding_buffer = None
        self.node_id_buffer = None
        self.workload_buffer = None
        self.timestep_buffer = None
        self.init_store()

    def init_store(self):
        self.embedding_store = [deque(maxlen=self.n_past) for _ in range(self.n_nodes)]
        for node in range(self.n_nodes):
            for _ in range(self.n_past):
                self.embedding_store[node].append(np.zeros(self.embedding_dim, dtype=np.float32))

    def add_embeddings(self, nodes, embeddings):
        for idx in range(len(nodes)):
            self.current_embedding_store[nodes[idx]] = embeddings[idx, :]

    def flush_embeddings_to_store(self, current_timestep, current_timestep_workloads):
        for node, embedding in self.current_embedding_store.items():
            self.embedding_store[node].append(embedding)
        self.current_embedding_store.clear()
        self._add_to_buffer(current_timestep, current_timestep_workloads)

    def _add_to_buffer(self, current_timestep, current_timestep_workloads):
        merged_embeddings = []

        for embeddings in self.embedding_store:
            merged_embeddings.append(np.vstack(tuple(embeddings)))

        merged_embeddings_np = np.array(merged_embeddings)
        shuffled_indices = np.random.permutation(merged_embeddings_np.shape[0])
        shuffled_embeddings = merged_embeddings_np[shuffled_indices, :, :]
        shuffled_workloads = current_timestep_workloads[shuffled_indices, :]
        current_timestep_values = np.full(self.n_nodes, current_timestep)

        if self.node_embedding_buffer is not None:
            self.node_embedding_buffer = np.concatenate(
                (self.node_embedding_buffer, shuffled_embeddings),
                axis=0
            )
        else:
            self.node_embedding_buffer = shuffled_embeddings

        if self.workload_buffer is not None:
            self.workload_buffer = np.concatenate((self.workload_buffer, shuffled_workloads))
        else:
            self.workload_buffer = shuffled_workloads

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

        embedding_batch = self.node_embedding_buffer[:self.buffer_size, :, :]
        nodes_batch = self.node_id_buffer[:self.buffer_size]
        workload_batch = self.workload_buffer[:self.buffer_size]
        timestep_batch = self.timestep_buffer[:self.buffer_size]

        self.node_embedding_buffer = self.node_embedding_buffer[self.buffer_size:, :, :]
        self.node_id_buffer = self.node_id_buffer[self.buffer_size:]
        self.workload_buffer = self.workload_buffer[self.buffer_size:]
        self.timestep_buffer = self.timestep_buffer[self.buffer_size:]

        return embedding_batch, nodes_batch, workload_batch, timestep_batch

    def reset_store(self):
        self.init_store()
        self.current_embedding_store.clear()
        self.node_embedding_buffer = None
        self.node_id_buffer = None
        self.workload_buffer = None
        self.timestep_buffer = None
