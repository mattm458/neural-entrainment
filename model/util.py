import torch
from torch import nn


class EmbeddingIterator:
    def __init__(self, embedding_seq, embedding_len, batch_size, device):
        self.embedding_seq = embedding_seq
        self.embedding_len = embedding_len
        self.batch_size = batch_size
        self.device = device

    def __iter__(self):
        self.next_embedding_start = torch.zeros(
            self.batch_size, dtype=torch.long, device=self.device
        )

        self.i = 0

        return self

    def __next__(self):
        next_embedding, next_embedding_len = self.peek()

        self.i += 1
        self.next_embedding_start += next_embedding_len

        return next_embedding, next_embedding_len

    def peek(self):
        next_embedding = []
        next_embedding_len = self.embedding_len[:, self.i]

        for batch_embeddings, start, length in zip(
            self.embedding_seq, self.next_embedding_start, next_embedding_len
        ):
            next_embedding.append(batch_embeddings[start : start + length])

        next_embedding = nn.utils.rnn.pad_sequence(
            next_embedding, batch_first=True, padding_value=1
        )

        return next_embedding, next_embedding_len
