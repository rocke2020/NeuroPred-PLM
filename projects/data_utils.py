import json
import logging
import math
import os
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

sys.path.append(os.path.abspath('.'))
from utils_comm.log_util import ic, logger


SEED = 0
random.seed(SEED)
np.random.seed(SEED)
train_file = 'dataset/train.csv'
test_file = 'dataset/test.csv'
SEQUENCE = "Sequence"
LEAST_SEQ_LENGTH = 4
MAX_SEQ_LEN = 50

amino_acids = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "L",
    "M",
    "N",
    "P",
    "K",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]


def is_natural_only_upper(seq):
    """If the char is not in upper case, treat as not natural"""
    if isinstance(seq, str) and seq:
        for aa in seq:
            if aa not in amino_acids:
                return False
        return True
    return False


def get_input_df_v1():
    df_test = pd.read_csv(test_file)
    df_train = pd.read_csv(train_file)
    df_test[SEQUENCE] = df_test['seq']
    df_train[SEQUENCE] = df_train['seq']
    df_test = df_test[df_test[SEQUENCE].apply(is_natural_only_upper)]
    df_train = df_train[df_train[SEQUENCE].apply(is_natural_only_upper)]
    df_test = df_test[df_test[SEQUENCE].apply(lambda x: len(x) <= MAX_SEQ_LEN)]
    df_train = df_train[df_train[SEQUENCE].apply(lambda x: len(x) <= MAX_SEQ_LEN)]
    df_test = df_test.drop_duplicates(subset=[SEQUENCE])
    df_train = df_train.drop_duplicates(subset=[SEQUENCE])
    ic(len(df_test), len(df_train))
    return df_train, df_test


if __name__ == '__main__':
    get_input_df_v1()
