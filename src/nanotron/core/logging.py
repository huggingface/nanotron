# coding=utf-8
# Copyright 2020 Optuna, Hugging Face
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Logging utilities. """
import logging
import os
import sys
from functools import lru_cache
from logging import (
    CRITICAL,
    DEBUG,
    ERROR,
    FATAL,
    INFO,
    NOTSET,
    WARNING,
    Logger,
)
from typing import Optional

from torch import distributed as torch_dist

from brrr.core import distributed as dist

log_levels = {
    "debug": DEBUG,
    "info": INFO,
    "warning": WARNING,
    "error": ERROR,
    "critical": CRITICAL,
    "fatal": FATAL,
    "notset": NOTSET,
}


class NewLineStreamHandler(logging.StreamHandler):
    """
    We want to apply formatter before each new line
    https://stackoverflow.com/a/38458877
    """

    def emit(self, record):
        lines = record.msg.split("\n")
        for line in lines:
            record.msg = line
            super().emit(record)


DEFAULT_HANDLER = NewLineStreamHandler()
DEFAULT_LOG_LEVEL = logging.WARNING
LIBRARY_NAME = __name__.split(".")[0]


def _get_default_logging_level():
    """
    If BRRR_LOGGING_LEVEL env var is set to one of the valid choices return that as the new default level. If it is
    not - fall back to ``_default_log_level``
    """
    env_level_str = os.getenv("BRRR_LOGGING_LEVEL", None)
    if env_level_str:
        if env_level_str in log_levels:
            return log_levels[env_level_str]
        else:
            logging.getLogger().warning(
                f"Unknown option BRRR_LOGGING_LEVEL={env_level_str}, "
                f"has to be one of: { ', '.join(log_levels.keys()) }"
            )
    return DEFAULT_LOG_LEVEL


def get_library_root_logger() -> Logger:
    return get_logger(LIBRARY_NAME)


def _configure_library_root_logger() -> None:
    library_root_logger = get_library_root_logger()
    library_root_logger.addHandler(DEFAULT_HANDLER)
    library_root_logger.setLevel(_get_default_logging_level())


def _reset_library_root_logger() -> None:
    library_root_logger = get_library_root_logger()
    library_root_logger.setLevel(logging.NOTSET)


def get_logger(name: Optional[str] = None) -> Logger:
    """
    Return a logger with the specified name.
    """
    logger_already_exists = isinstance(logging.root.manager.loggerDict.get(name, None), Logger)
    logger = logging.getLogger(name)

    if logger_already_exists or name is None:
        # if name is None we return root logger
        return logger

    # If the logger is in a `brrr` module then we remove the capability to propagate
    if LIBRARY_NAME == name.split(".", 1)[0]:
        if LEVEL is not None:
            logger.setLevel(LEVEL)
        else:
            logger.setLevel(_get_default_logging_level())
        if HANDLER is not None:
            logger.handlers.clear()
            logger.addHandler(HANDLER)

        logger.propagate = False

    return logger


def get_verbosity() -> int:
    """
    Return the current level for the BRRR root logger as an int.
    Returns:
        :obj:`int`: The logging level.
    .. note::
        BRRR has following logging levels:
        - 50: ``brrr.logging.CRITICAL`` or ``brrr.logging.FATAL``
        - 40: ``brrr.logging.ERROR``
        - 30: ``brrr.logging.WARNING`` or ``brrr.logging.WARN``
        - 20: ``brrr.logging.INFO``
        - 10: ``brrr.logging.DEBUG``
    """

    return get_library_root_logger().getEffectiveLevel()


LEVEL = None


def set_verbosity(verbosity: int) -> None:
    """
    Set the verbosity level for the all `brrr` loggers.

    Args:
        verbosity (:obj:`int`):
            Logging level, e.g., one of:
            - ``brrr.logging.CRITICAL`` or ``brrr.logging.FATAL``
            - ``brrr.logging.ERROR``
            - ``brrr.logging.WARNING`` or ``brrr.logging.WARN``
            - ``brrr.logging.INFO``
            - ``brrr.logging.DEBUG``
    """
    all_brrr_loggers = {
        name: logger
        for name, logger in logging.root.manager.loggerDict.items()
        if isinstance(logger, Logger) and (name.startswith(f"{LIBRARY_NAME}.") or name == LIBRARY_NAME)
    }
    for name, logger in all_brrr_loggers.items():
        logger.setLevel(verbosity)

        # We update all handles to be at the current verbosity as well.
        for handle in logger.handlers:
            handle.setLevel(verbosity)

    global LEVEL
    LEVEL = verbosity


HANDLER = None


def set_formatter(formatter: logging.Formatter) -> None:
    """
    Set a new custom formatter as the current handler.
    Note: it's important to first set level and then

    :param formatter:
    :return:
    """
    handler = NewLineStreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(get_verbosity())
    handler.flush = sys.stderr.flush

    all_brrr_loggers = {
        name: logger
        for name, logger in logging.root.manager.loggerDict.items()
        if isinstance(logger, Logger) and (name.startswith(f"{LIBRARY_NAME}.") or name == LIBRARY_NAME)
    }
    for name, logger in all_brrr_loggers.items():
        # We keep only a single handler
        logger.handlers.clear()
        logger.addHandler(handler)

    global HANDLER
    HANDLER = handler


def log_rank(
    msg: str,
    logger: Logger,
    level: int,
    group: Optional[dist.ProcessGroup] = None,
    rank: Optional[int] = None,
    **kwargs,
):
    # Use default group is group is not provided
    if group is None:
        group = torch_dist.distributed_c10d._get_default_group()

    # rank is None means everyone logs
    if rank is None or dist.get_rank(group) == rank:
        logger.log(level, msg, **kwargs)


@lru_cache(maxsize=None)
def warn_once(
    logger: Logger, msg: str, group: Optional[dist.ProcessGroup] = None, rank: Optional[int] = None, **kwargs
):
    log_rank(msg=msg, logger=logger, level=logging.WARNING, group=group, rank=rank, **kwargs)


_configure_library_root_logger()
