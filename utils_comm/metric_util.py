import logging

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger()
logging.basicConfig(
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M",
    format="%(asctime)s %(filename)s %(lineno)d: %(message)s",
)


def calc_metrics(y_true, y_score, threshold=0.5):
    """NB: return order is accuracy, f1, mcc, precision, recall
    -
    if threshold > 0: y_score = y_score >= threshold; else, directly uses y_score, that's treat y_score as integer
    """
    if threshold > 0:
        if not isinstance(y_score, np.ndarray):
            y_score = np.array(y_score)
        y_pred_id = y_score >= threshold
    else:
        y_pred_id = y_score
    accuracy = accuracy_score(y_true, y_pred_id)
    f1 = f1_score(y_true, y_pred_id)
    mcc = matthews_corrcoef(y_true, y_pred_id)
    precision = precision_score(y_true, y_pred_id)
    recall = recall_score(y_true, y_pred_id)
    return accuracy, f1, mcc, precision, recall


def calc_f1_precision_recall(y_true, y_predict):
    """ """
    # accuracy = accuracy_score(y_true, y_predict)
    f1 = f1_score(y_true, y_predict)
    precision = precision_score(y_true, y_predict)
    recall = recall_score(y_true, y_predict)
    return f1, precision, recall


def find_threshold(y_true, y_score, alpha=0.05):
    """return threshold when fpr <= 0.05"""
    fpr, tpr, thresh = roc_curve(y_true, y_score)
    for i, _fpr in enumerate(fpr):
        if _fpr > alpha:
            return thresh[i - 1]


def roc(y_true, y_score):
    fpr, tpr, thresh = roc_curve(y_true, y_score)
    roc = roc_auc_score(y_true, y_score)
    return roc, fpr, tpr


def calc_metrics_at_thresholds(
    y_true, y_pred_probability, thresholds=None, default_threshold=None
):
    """ """
    performances = {}
    if default_threshold:
        # logger.info('calc_metrics at threshold %s', default_threshold)
        accuracy, f1, mcc, precision, recall = calc_metrics(
            y_true, y_pred_probability, default_threshold
        )
        performances[f"default_threshold_{default_threshold}"] = {
            "accuracy": accuracy,
            "f1": f1,
            "mcc": mcc,
            "precision": precision,
            "recall": recall,
        }

    if not thresholds:
        thresholds = []
    thresholds = sorted(thresholds, reverse=True)
    if 0.5 not in thresholds:
        thresholds.append(0.5)
    for threshold in thresholds:
        # logger.info('calc_metrics at threshold %s', threshold)
        accuracy, f1, mcc, precision, recall = calc_metrics(
            y_true, y_pred_probability, threshold
        )
        performances[f"threshold_{threshold}"] = {
            "accuracy": accuracy,
            "f1": f1,
            "mcc": mcc,
            "precision": precision,
            "recall": recall,
        }
    return performances


def jensen_shannon_distance(p, q):
    """
    method to compute the Jenson-Shannon Distance
    between two probability distributions
    """
    # convert the vectors into numpy arrays in case that they aren't
    p = np.array(p)
    q = np.array(q)

    # calculate m
    m = (p + q) / 2

    # compute Jensen Shannon Divergence
    divergence = (stats.entropy(p, m) + stats.entropy(q, m)) / 2  # type: ignore

    # compute the Jensen Shannon Distance
    distance = np.sqrt(divergence)

    return distance


def calc_spearmanr(x, y, notes=""):
    """ """
    res = stats.spearmanr(x, y)
    logger.info(f"{notes} spearmanr: {res}")
    if hasattr(res, "correlation"):
        spearman_ratio = float(res.correlation)  # type: ignore
    else:
        assert hasattr(res, "statistic")
        spearman_ratio = float(res.statistic)  # type: ignore
    if hasattr(res, "pvalue"):
        pvalue = float(res.pvalue)  # type: ignore
        logger.info(f"{notes} spearmanr pvalue: {pvalue}")
    logger.info(f"{notes} spearman_ratio: {spearman_ratio}")
    return spearman_ratio
