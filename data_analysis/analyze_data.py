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
SEQUENCE = "Sequence"
random.seed(SEED)
np.random.seed(SEED)
train_file = 'dataset/train.csv'
test_file = 'dataset/test.csv'
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


def basic_check():
    """ Bad data, test and train have 4 seq shared.

    len(seq_test_unique): 888
    len(seq_train_unique): 8003

    label      seq_len
    count  8887.000000  8887.000000
    mean      0.497806    22.693485
    std       0.500023    18.191652
    min       0.000000     6.000000
    25%       0.000000    11.000000
    50%       0.000000    17.000000
    75%       1.000000    28.000000
    max       1.000000    99.000000
    ic| analyze_data.py:86 in basic_check()- natural len(df): 8699

    """
    df_test = pd.read_csv(test_file)
    df_train = pd.read_csv(train_file)
    df_test[SEQUENCE] = df_test['seq']
    df_train[SEQUENCE] = df_train['seq']

    seq_test = df_test[SEQUENCE].tolist()
    seq_train = df_train[SEQUENCE].tolist()
    seq_test_unique = set(seq_test)
    seq_train_unique = set(seq_train)
    ic(len(seq_test), len(seq_train), len(seq_test_unique), len(seq_train_unique))
    cross_seqs1 = set(seq_test) - set(seq_train)
    cross_seqs2 = set(seq_train) - set(seq_test)
    ic(len(cross_seqs1), len(cross_seqs2))
    df = pd.concat([df_train, df_test])
    ic(len(df))
    df = df.drop_duplicates(subset=[SEQUENCE])
    ic(len(df))
    df['seq_len'] = df[SEQUENCE].apply(lambda x: len(x))
    logger.info('%s', df.describe())
    df = df[df['seq'].map(is_natural_only_upper)]
    ic(len(df))


def group_seq():
    """ Group seqs to check the possible label num error. """
    df_test = pd.read_csv(test_file)
    df_train = pd.read_csv(train_file)
    df_test[SEQUENCE] = df_test['seq']
    df_train[SEQUENCE] = df_train['seq']
    df_train = df_train.groupby(SEQUENCE).agg(set).reset_index(drop=False)
    df_test = df_test.groupby(SEQUENCE).agg(set).reset_index(drop=False)
    df_train['label_num'] = df_train['label'].apply(lambda x: len(x))
    df_test['label_num'] = df_test['label'].apply(lambda x: len(x))
    # check contraversal labels, no contraversal labels.
    df_train_ = df_train[df_train['label_num'] > 1]
    df_test_ = df_test[df_test['label_num'] > 1]
    ic(len(df_train_), len(df_test_))
    ic(len(df_train), len(df_test))


if __name__ == '__main__':
    basic_check()
    # group_seq()
