from functools import cache
import torch, os, sys

sys.path.append(os.path.abspath("."))
import esm
from torch.nn.utils.rnn import pad_sequence
import time
from utils_comm.log_util import logger
from utils_comm.train_util import get_device
import torch.nn as nn
from typing import Sequence, List, Tuple
from copy import deepcopy
from torch import Tensor
from .vocabulary import Vocabulary


class Dataset(torch.utils.data.Dataset):
    """ """

    def __init__(
        self, df, vocabulary: Vocabulary, category_key="", seq_col_name="Sequence"
    ):
        self.vocabulary = vocabulary
        self.df = df
        self.category_key = category_key
        self.seq_col_name = seq_col_name

    def __getitem__(self, i):
        seq_str = self.df.iloc[i][self.seq_col_name]
        tensor = self.vocabulary.seq_to_tensor_no_start(seq_str)
        if self.category_key:
            category = self.df.iloc[i][self.category_key]
            return tensor, category
        else:
            return tensor

    def __len__(self):
        return len(self.df)


class DatasetPredict(torch.utils.data.Dataset):
    """Input seqs and without category"""

    def __init__(self, seqs: list[str], vocabulary: Vocabulary):
        self.vocabulary = vocabulary
        self.seqs = seqs

    def __getitem__(self, i):
        seq_str = self.seqs[i]
        tensor = self.vocabulary.seq_to_tensor(seq_str)
        return tensor

    def __len__(self):
        return len(self.seqs)


def collate_fn_rnn(arr):
    """Function to take a list of encoded sequences and turn them into a batch"""
    arr_seq = [element[0] for element in arr]
    cat_seq = [element[1] for element in arr]
    packed_seq = torch.nn.utils.rnn.pack_sequence(arr_seq, enforce_sorted=False)
    seq_unpacked, lens_unpacked = torch.nn.utils.rnn.pad_packed_sequence(
        packed_seq, batch_first=True
    )
    return seq_unpacked, lens_unpacked, torch.tensor(cat_seq)


def collate_fn_cnn(arr, total_length=52):
    """Function to take a list of encoded sequences and turn them into a batch, vocabulary adds start and end."""
    arr_seq = [element[0] for element in arr]
    cat_seq = [element[1] for element in arr]
    packed_seq = torch.nn.utils.rnn.pack_sequence(arr_seq, enforce_sorted=False)
    seq_unpacked, lens_unpacked = torch.nn.utils.rnn.pad_packed_sequence(
        packed_seq, batch_first=True, total_length=total_length
    )
    return seq_unpacked, lens_unpacked, torch.tensor(cat_seq)


def collate_fn_pretrain_rnn(arr):
    """ """
    seq_emb_tensors = [element[0] for element in arr]
    lengths = [element[1] for element in arr]
    categories = [element[2] for element in arr]
    padded_seq = pad_sequence(seq_emb_tensors, batch_first=True)
    return padded_seq, lengths, categories


def collate_fn_pretrain_cnn(arr):
    """ """
    padded_seq = [element[0] for element in arr]
    lengths = [element[1] for element in arr]
    categories = [element[2] for element in arr]
    return padded_seq, lengths, categories


def collate_fn_no_category(arr) -> Tuple[Tensor, Tensor]:
    """Function to take a list of encoded sequences and turn them into a batch
    Args:
        arr: list[tensor], inside tensor shape: seq-len * dim-size
    Returns:
        seq_unpacked, lens_unpacked
    """
    packed_seq = torch.nn.utils.rnn.pack_sequence(arr, enforce_sorted=False)
    padded_seq = torch.nn.utils.rnn.pad_packed_sequence(packed_seq, batch_first=True)
    return padded_seq


def collate_fn_cnn_no_category(arr):
    """Function to take a list of encoded sequences and turn them into a batch"""
    padded_seq = torch.stack(arr).unsqueeze(dim=1)
    # logger.info(f'padded_seq.shape {padded_seq.shape}')
    return padded_seq


if __name__ == "__main__":
    seqs = [
        "ADE",
        "ADaAX",
        "RHZ",
    ]
