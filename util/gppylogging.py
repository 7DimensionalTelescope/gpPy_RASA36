# gppylogging.py

import logging
import logging.handlers
import os
import sys
import traceback
from multiprocessing import Queue as MPQueue

_listener: 'logging.handlers.QueueListener | None' = None
_log_queue: 'MPQueue | None' = None


def setup_global_logging(logdir, obs, obs_date, log_level=logging.DEBUG, obs_subdir=None):
    """
    Set up root logger with QueueHandler/QueueListener for safe multi-process use.

    The QueueListener thread in the main process is the only writer to the log file.
    Forked worker processes inherit a QueueHandler that sends records through a
    multiprocessing.Queue, eliminating concurrent-write collisions.

    With obs_subdir (Option C): log → {logdir}/{obs}/{obs_subdir}/{obs_date}.log
    Without obs_subdir (legacy): log → {logdir}/{obs}/{obs}_{obs_date}.log
    """
    global _listener, _log_queue

    def global_exception_handler(exctype, value, tb):
        full_traceback = ''.join(traceback.format_exception(exctype, value, tb))
        logging.critical(f"Uncaught Exception:\n{full_traceback}")
        print(f"Critical Error: {value}. Check log for details.")

    if obs_subdir is not None:
        log_dir = os.path.join(logdir, obs.upper(), obs_subdir)
        log_file = os.path.join(log_dir, f'{obs_date}.log')
    else:
        log_dir = os.path.join(logdir, obs.upper())
        log_file = os.path.join(log_dir, f'{obs.upper()}_{obs_date}.log')
    os.makedirs(log_dir, exist_ok=True)

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    ))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))

    _log_queue = MPQueue(-1)

    if _listener is not None:
        _listener.stop()
    _listener = logging.handlers.QueueListener(
        _log_queue, file_handler, console_handler, respect_handler_level=True
    )
    _listener.start()

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(logging.handlers.QueueHandler(_log_queue))
    root.setLevel(log_level)

    sys.excepthook = global_exception_handler

    logging.info(f"Logging initialized for {obs.upper()} on {obs_date}")

    return root


def setup_worker_logging(log_queue):
    """Replace inherited root-logger handlers with a QueueHandler in a forked worker."""
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(logging.handlers.QueueHandler(log_queue))
    root.setLevel(logging.DEBUG)


def get_log_queue():
    return _log_queue


def stop_listener():
    global _listener
    if _listener is not None:
        _listener.stop()
        _listener = None
