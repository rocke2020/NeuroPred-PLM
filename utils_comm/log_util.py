""" Basic usage example: 

from utils_comm.log_util import logger

logger.info('comments')
logger.info(f'df.shape {df.shape}')
logger.info('df.shape %s', df.shape)
"""

import contextlib
import logging
import os
import time

from icecream import ic
from pandas import DataFrame

ic.configureOutput(includeContext=True, argToStringFunction=str)
ic.lineWrapWidth = 120


FMT = '%(asctime)s %(filename)s %(lineno)d: %(message)s'
DATE_FORMAT = '%y-%m-%d %H:%M:%S'


def get_logger(name=None, log_file=None, log_level=logging.INFO):
    """ default log level DEBUG """
    _logger = logging.getLogger(name)
    logging.basicConfig(format=FMT, datefmt=DATE_FORMAT)
    if log_file is not None:
        log_file_folder = os.path.split(log_file)[0]
        if log_file_folder:
            os.makedirs(log_file_folder, exist_ok=True)
        fh = logging.FileHandler(log_file, 'w', encoding='utf-8')
        fh.setFormatter(logging.Formatter(FMT, DATE_FORMAT))
        _logger.addHandler(fh)
    _logger.setLevel(log_level)
    return _logger


logger = get_logger()


def log_df_basic_info(df: DataFrame, log_func=None, comments='', full_info=False,
                      partial_col_num=4):
    """ For large df, describe() use much more time! """
    if log_func is None:
        log_func = logger
    if comments:
        log_func.info(f'comments {comments}')
    log_func.info(f'df.shape {df.shape}')
    columns = df.columns.to_list()
    if full_info:
        log_func.info(f'df.columns {columns}')
        log_func.info(f'df.head()\n{df.head()}')
        log_func.info(f'df.tail()\n{df.tail()}')
        log_func.info(f'df.describe()\n{df.describe()}')
    else:
        log_func.info(f'df.columns [:{partial_col_num}] {columns[:partial_col_num]}')
        log_func.info(f'df.columns [-partial_col_num:] {columns[-partial_col_num:]}')
        log_func.info(f'df[columns[:partial_col_num]].head() {df[columns[:partial_col_num]].head()}')
        log_func.info(f'df[columns[-partial_col_num:]].tail() {df[columns[-partial_col_num:]].tail()}')
        if len(df) <= 200_000:
            log_func.info(f'df[columns[:partial_col_num]].describe()\n{df[columns[:partial_col_num]].describe()}')


@contextlib.contextmanager
def timing(msg: str = ''):
    logging.info('Started %s', msg)
    tic = time.time()
    yield
    toc = time.time()
    logging.info('Finished %s in %.3f seconds', msg, toc - tic)


if __name__ == "__main__":
    @timing('start')
    def a():
        """  """
        print(1)

    a()
