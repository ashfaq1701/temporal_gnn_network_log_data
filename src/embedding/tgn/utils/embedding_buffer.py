from collections import deque

import numpy as np


class EmbeddingBuffer:
    def __init__(self, n_past, n_nodes, embedding_dim):
        self.n_past = n_past
        self.n_nodes = n_nodes
        self.embedding_dim = embedding_dim
        self.embedding_store = []
        self.current_embedding_store = {}
        self.init_store()

    def init_store(self):
        self.embedding_store = [deque(maxlen=self.n_past) for _ in range(self.n_nodes)]
        for node in range(self.n_nodes):
            for _ in range(self.n_past):
                self.embedding_store[node].append(np.zeros(self.embedding_dim, dtype=np.float32))

    def add_embeddings(self, nodes, embeddings):
        for idx in range(len(nodes)):
            self.current_embedding_store[nodes[idx]] = embeddings[idx, :]

    def flush_embeddings_to_store(self):
        for node, embedding in self.current_embedding_store.items():
            self.embedding_store[node].append(embedding)
        self.current_embedding_store.clear()

    def reset_store(self):
        self.init_store()
        self.current_embedding_store.clear()
