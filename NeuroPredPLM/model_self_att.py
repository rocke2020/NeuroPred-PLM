import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import os
from transformers.activations import GELUActivation

# from einops import rearrange

from .utils import length_to_mask_pad_as_true
from .vocabulary import Vocabulary


class BertEmbeddings(nn.Module):
    """
    docstring
    """
    def __init__(
        self,
        hidden_size=768,
        vocab_size=23,
        max_len=52,
        hidden_dropout_prob=0.1,
    ):
        super().__init__()
        self.vocab_embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_len, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.register_buffer(
            "position_ids", torch.arange(max_len).expand((1, -1)), persistent=False
        )

    def forward(self, x):
        embeddings = self.vocab_embedding(x)
        batch_seq_length = embeddings.shape[1]
        position_ids = self.position_ids[:, :batch_seq_length]
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertIntermediate(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.intermediate_act_fn = GELUActivation()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob=0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertSelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob=0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, hidden_size=768, head=12, hidden_dropout_prob=0.1):
        super().__init__()
        self.att = nn.MultiheadAttention(hidden_size, head, batch_first=True)
        self.output = BertSelfOutput(hidden_size, hidden_dropout_prob)

    def forward(self, hidden_states, seq_lengths):
        mask = length_to_mask_pad_as_true(seq_lengths).to(
            hidden_states.device
        )
        attention_output, attn_output_weights = self.att(
            hidden_states, hidden_states, hidden_states, key_padding_mask=mask
        )
        result = self.output(attention_output, hidden_states)
        return result, attn_output_weights


class BertLayer(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        self.attention = BertAttention()
        self.intermediate = BertIntermediate(hidden_size)
        self.output = BertOutput(hidden_size)

    def forward(self, hidden_states, seq_lengths):
        attention_output, attn_output_weights = self.attention(hidden_states, seq_lengths)
        intermediate_output = self.intermediate(attention_output)
        result = self.output(intermediate_output, attention_output)
        return result


class BertEncoder(nn.Module):
    def __init__(self, num_hidden_layers=1):
        super().__init__()
        self.layer = nn.ModuleList([BertLayer() for _ in range(num_hidden_layers)])

    def forward(self, hidden_states, seq_lengths):
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, seq_lengths)
        return hidden_states


class SimpleBert(nn.Module):
    def __init__(
        self,
        hidden_size=768,
        vocab_size=23,
        max_len=52,
        head=12,
        num_labels=2,
        hidden_dropout_prob=0.1,
    ):
        super().__init__()
        self.embeddings = BertEmbeddings(
            hidden_size, vocab_size, max_len, hidden_dropout_prob
        )
        self.encoder = BertEncoder()

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, x, seq_lengths):
        embeddings = self.embeddings(x)
        encoder_outputs = self.encoder(embeddings, seq_lengths)
        encoder_outputs = torch.mean(encoder_outputs, dim=1)
        result = self.fc(encoder_outputs)
        return result
