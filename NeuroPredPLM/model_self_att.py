import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import os

# from einops import rearrange

from .utils import length_to_mask_pad_as_true
from .vocabulary import Vocabulary


class SelfAtt(nn.Module):
    def __init__(
        self,
        embed_dim=1280,
        hidden_size=640,
        vocab_size=23,
        max_len=52,
        head=12,
        num_labels=2,
    ):
        super().__init__()
        self.vocab_embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = nn.Embedding(max_len, embed_dim)
        self.embed_layer_norm = nn.LayerNorm(embed_dim)
        self.embed_dropout = nn.Dropout(0.1)

        self.att = nn.MultiheadAttention(embed_dim, head, batch_first=True)
        self.fcn = nn.Sequential(
            nn.Linear(embed_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_labels),
        )
        self.register_buffer(
            "position_ids", torch.arange(max_len).expand((1, -1)), persistent=False
        )

    def forward(self, x, seq_lengths):
        embeddings = self.vocab_embedding(x)
        batch_seq_length = embeddings.shape[1]
        position_ids = self.position_ids[:, : batch_seq_length]
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = embeddings + position_embeddings
        embeddings = self.embed_layer_norm(embeddings)
        embeddings = self.embed_dropout(embeddings)

        mask = length_to_mask_pad_as_true(torch.tensor(seq_lengths)).to(x.device)
        att, attn_output_weights = self.att(
            inputs, inputs, inputs, key_padding_mask=mask
        )
        att = torch.mean(att, dim=1)
        result = self.fcn(att)
        return result, attn_output_weights
