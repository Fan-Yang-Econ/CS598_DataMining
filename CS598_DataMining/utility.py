import os
import numpy as np
import typing
import logging
import sys
import traceback


def add_end_backslash(folder_path):
    """

    :param folder_path:
    :return:

    add_end_backslash('/tmp')
    """
    if folder_path[-1] != '/':
        folder_path = folder_path + '/'
    return folder_path


def exception_handler(type, value, tb):
    """
    Enable to also log the exceptions

    https://stackoverflow.com/questions/8050775/using-pythons-logging-module-to-log-all-exceptions-and-errors

    :param type:
    :param value:
    :param tb:
    :return:
    """
    logging.error(f"Uncaught Exception Found: \n{type}: {value}")
    logging.info(f'TRACK BACK:')
    for i in traceback.extract_tb(tb).format():
        logging.info(i)
    
    # run the original handler
    sys.__excepthook__(type, value, tb)


def create_folder(folder_path=None, mode=0o777):
    if os.path.isdir(folder_path) is False:
        os.makedirs(folder_path, mode=mode)
        logging.debug(f"{folder_path} was created")
    
    return add_end_backslash(folder_path)


def cal_cosine_simi(w1_vector, w2_vector, max_digits=3):
    _n = np.dot(w1_vector, w2_vector) / \
         sum([i ** 2 for i in w1_vector]) ** 0.5 / \
         sum([i ** 2 for i in w2_vector]) ** 0.5
    
    try:
        return round(_n * 10 ** max_digits) / 10 ** max_digits
    except Exception:
        return _n


def cal_avg_vt(vt_list):
    vt_sum = None
    for vt in vt_list.tolist():
        # len(vt)
        # len(vt_sum)
        if not vt_sum:
            vt_sum = vt
        else:
            print(len(vt_sum))
            vt_sum = [i + vt[index_i] for index_i, i in enumerate(vt_sum)]
    
    vt_avg = [int(i / len(vt_list) * 1000) / 1000 for i in vt_sum]
    
    return vt_avg


def set_logging(level=20,
                path: typing.Optional[str] = None,
                log_format='%(asctime)s:%(name)s-%(funcName)s: %(message)s',  # use "%(message)s" simpler format for lambda logging
                datefmt='%Y-%m-%d %H:%M:%S'
                ):
    """

    :param level:
    :param path:
        If provided, we will also write the file to the local path

    :param log_format:
        '%(levelname)s-%(name)s-%(funcName)s:\n %(message)s'
    :return:
    """
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    if path:
        logging.basicConfig(level=level, format=log_format,
                            # Two handlers, log to both console and the file.
                            handlers=[logging.FileHandler(filename=path), logging.StreamHandler()],
                            datefmt=datefmt)
    else:
        logging.basicConfig(level=level, format=log_format, datefmt=datefmt)
    
    # Also log exceptions
    sys.excepthook = exception_handler
