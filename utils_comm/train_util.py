import logging
import random

import numpy as np
import torch
import torch.optim as optim
from pandas import DataFrame
from sklearn.model_selection import train_test_split

logger = logging.getLogger()
logging.basicConfig(
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M",
    format="%(asctime)s %(filename)s %(lineno)d: %(message)s",
)
SEQUENCE = "Sequence"


def set_seed(seed: int = 0):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`): The seed to set.
    """
    logger.info('seed %s', seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    except Exception as identifier:
        pass

def get_device(device_id=0):
    """if device_id < 0: device = "cpu" """
    gpu_count = 0
    if device_id < 0:
        device = "cpu"
    elif torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count > device_id:
            device = f"cuda:{device_id}"
        else:
            device = "cuda:0"
    else:
        device = "cpu"
    logger.info(f"gpu_count {gpu_count}, device: {device}")
    return device


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if self.warmup > 0 and epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


def nan_equal(a, b):
    try:
        np.testing.assert_equal(a, b)
    except AssertionError:
        return False
    return True


def models_are_equal(model1, model2):
    model1.vocabulary == model2.vocabulary
    model1.hidden_size == model2.hidden_size
    for a, b in zip(model1.model.parameters(), model2.model.parameters()):
        if nan_equal(a.detach().numpy(), b.detach().numpy()) == True:
            logger.info("true")


def run_compile_when_pt2(model, enable_complile=True, save_gpu_memory=True):
    """ """
    pt_version = torch.__version__.split(".")[0]
    if pt_version == "2" and enable_complile:
        if save_gpu_memory:
            _model = torch.compile(model)
        else:
            _model = torch.compile(model, mode="reduce-overhead")
    else:
        _model = model
    return _model


def get_weights_from_df(df, category_label_column, prefix="test"):
    """
    Returns:
        Weights [positive_num/negative_num, 1],  is the reversed ratio,
        weights[0] for neg, weights[1] for pos.
    """
    negative_num = len(df[df[category_label_column] == 0])
    positive_num = len(df[df[category_label_column] == 1])
    # Weights is the reversed ratio, weights[0] for neg, weights[0] for pos.
    weights = [positive_num / negative_num, 1]
    logger.info(
        f"{prefix} data positive {positive_num} vs negative {negative_num}, ratio {weights}"
    )
    return weights


def split_train_test_in_df(
    df: DataFrame,
    binary_label_column,
    test_size=0.2,
    random_seed=1,
    use_weight=True,
    seq_column=SEQUENCE,
):
    """
    Returns:
        df_train, df_test, weights
        weights [positive_num/negative_num, 1],  is the reversed ratio, weights[0] for neg, weights[1] for pos.
    """
    labels = df[binary_label_column]
    df_train, df_test = train_test_split(
        df, test_size=test_size, stratify=labels, random_state=random_seed
    )
    log_df_train_test_basic_info(df, df_train, df_test, seq_col=seq_column)
    get_weights_from_df(df, category_label_column=binary_label_column, prefix="all")
    if use_weight:
        weights = get_weights_from_df(
            df_train, category_label_column=binary_label_column, prefix="train"
        )
    else:
        weights = None
    return df_train, df_test, weights


def log_df_train_test_basic_info(
    df, df_train, df_test, seq_col=SEQUENCE, binary_label_col=""
):
    """ """
    logger.info(
        "All data num %s len(df_train) %s, len(df_test) %s",
        len(df),
        len(df_train),
        len(df_test),
    )
    if "len" not in df.columns:
        df["len"] = df[seq_col].apply(len)
    logger.info("max len df %s, min len df %s", max(df["len"]), min(df["len"]))
    logger.info(f"df_train.columns [:7] {df_train.columns.tolist()[:7]}")
    logger.info("df_train seq.head()\n%s", df_train[seq_col].head())
    logger.info(f"df_test seq.head()\n{df_test[seq_col].head()}")
    if binary_label_col:
        if binary_label_col in df:
            pos_df = df[df[binary_label_col] == 1]
            neg_df = df[df[binary_label_col] == 0]
            logger.info("pos_df num %s, neg_df num %s", len(pos_df), len(neg_df))
        else:
            logger.warning(
                f"binary_label_col {binary_label_col} not in the train df columns."
            )


if __name__ == "__main__":
    get_device()
