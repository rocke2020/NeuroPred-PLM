import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from icecream import ic
from pandas import DataFrame
from torch import nn
from torch.utils import data
from tqdm import tqdm
sys.path.append(os.path.abspath('.'))
from utils_comm.log_util import logger
from utils_comm.train_util import set_seed, get_device
from projects.data_utils import get_input_df_v1
from NeuroPredPLM.vocabulary import Vocabulary


class SelfAttTrainer():
    """
    docstring
    """
    def __init__(self) -> None:
        self.device = get_device()
        set_seed()
        self.df_train, self.df_test = get_input_df_v1()
        self.embedding_type = "vocab"
        self.fixed_vocab_file = 'config/classifier_vocab.json'

    def get_dataset(self):
        vocabulary = Vocabulary.load_vocab_file(self.fixed_vocab_file)
        training_dataset = Dataset(df_train, vocabulary, self.category_col_name)
        test_dataset = Dataset(df_test, vocabulary, self.category_col_name)
        
if __name__ == '__main__':
    main()
