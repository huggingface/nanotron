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
from dataclasses import dataclass
from functools import lru_cache
from logging import CRITICAL, DEBUG, ERROR, FATAL, INFO, NOTSET, WARNING, Formatter, Logger
from typing import List, Optional, Union

import torch
from torch import distributed as torch_dist

from nanotron import distributed as dist
from nanotron.parallel import ParallelContext

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
    If NANOTRON_LOGGING_LEVEL env var is set to one of the valid choices return that as the new default level. If it is
    not - fall back to ``_default_log_level``
    """
    env_level_str = os.getenv("NANOTRON_LOGGING_LEVEL", None)
    if env_level_str:
        if env_level_str in log_levels:
            return log_levels[env_level_str]
        else:
            logging.getLogger().warning(
                f"Unknown option NANOTRON_LOGGING_LEVEL={env_level_str}, "
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


def get_logger(name: Optional[str] = None, log_level: Optional[str] = None) -> Logger:
    """
    Return a logger with the specified name.
    """
    logger_already_exists = isinstance(logging.root.manager.loggerDict.get(name, None), Logger)
    logger = logging.getLogger(name)

    if logger_already_exists or name is None:
        # if name is None we return root logger
        return logger

    # If the logger is in a `nanotron` module then we remove the capability to propagate
    if LIBRARY_NAME == name.split(".", 1)[0]:
        if log_level is not None:
            logger.setLevel(log_level.upper())
        elif LEVEL is not None:
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
    Return the current level for the Nanotron root logger as an int.
    Returns:
        :obj:`int`: The logging level.
    .. note::
        Nanotron has following logging levels:
        - 50: ``nanotron.logging.CRITICAL`` or ``nanotron.logging.FATAL``
        - 40: ``nanotron.logging.ERROR``
        - 30: ``nanotron.logging.WARNING`` or ``nanotron.logging.WARN``
        - 20: ``nanotron.logging.INFO``
        - 10: ``nanotron.logging.DEBUG``
    """

    return get_library_root_logger().getEffectiveLevel()


LEVEL = None


def set_verbosity(verbosity: int) -> None:
    """
    Set the verbosity level for the all `nanotron` loggers.

    Args:
        verbosity (:obj:`int`):
            Logging level, e.g., one of:
            - ``nanotron.logging.CRITICAL`` or ``nanotron.logging.FATAL``
            - ``nanotron.logging.ERROR``
            - ``nanotron.logging.WARNING`` or ``nanotron.logging.WARN``
            - ``nanotron.logging.INFO``
            - ``nanotron.logging.DEBUG``
    """
    all_nanotron_loggers = {
        name: logger
        for name, logger in logging.root.manager.loggerDict.items()
        if isinstance(logger, Logger) and (name.startswith(f"{LIBRARY_NAME}.") or name == LIBRARY_NAME)
    }
    for name, logger in all_nanotron_loggers.items():
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

    all_nanotron_loggers = {
        name: logger
        for name, logger in logging.root.manager.loggerDict.items()
        if isinstance(logger, Logger) and (name.startswith(f"{LIBRARY_NAME}.") or name == LIBRARY_NAME)
    }
    for name, logger in all_nanotron_loggers.items():
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
    """Log only if the current process is the rank specified."""
    # Use default group is group is not provided
    if group is None:
        group = torch_dist.distributed_c10d._get_default_group()

    # rank is None means everyone logs
    if rank is None or dist.get_rank(group) == rank:
        logger.log(level, msg, **kwargs)


@lru_cache(maxsize=None)
def warn_once(
    msg: str, logger: Logger, group: Optional[dist.ProcessGroup] = None, rank: Optional[int] = None, **kwargs
):
    log_rank(msg=msg, logger=logger, level=logging.WARNING, group=group, rank=rank, **kwargs)


def human_format(num: float, billions: bool = False, divide_by_1024: bool = False) -> str:
    if abs(num) < 1:
        return "{:.3g}".format(num)
    SIZES = ["", "K", "M", "G", "T", "P", "E"]
    num = float("{:.3g}".format(num))
    magnitude = 0
    i = 0
    while abs(num) >= 1000 and i < len(SIZES) - 1:
        magnitude += 1
        num /= 1000.0 if not divide_by_1024 else 1024.0
        i += 1
    return "{}{}".format("{:f}".format(num).rstrip("0").rstrip("."), SIZES[magnitude])


def log_memory(logger: logging.Logger):
    log_rank(
        f" Memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f}MiB."
        f" Peak allocated {torch.cuda.max_memory_allocated() / 1024**2:.2f}MiB."
        f" Peak reserved: {torch.cuda.max_memory_reserved() / 1024**2:.2f}MiB",
        logger=logger,
        level=logging.INFO,
        rank=0,
    )
    torch.cuda.reset_peak_memory_stats()


@dataclass
class LogItem:
    tag: str
    scalar_value: Union[float, int, str]
    log_format: Optional[str] = None


@dataclass
class LoggerWriter:
    global_step: int

    def add_scalar(self, tag: str, scalar_value: Union[float, int], log_format=None) -> str:
        if log_format == "human_format":
            log_str = f"{tag}: {human_format(scalar_value)}"
        else:
            log_str = f"{tag}: {scalar_value:{log_format}}" if log_format is not None else f"{tag}: {scalar_value}"
        return log_str

    def add_scalars_from_list(self, log_entries: List[LogItem], iteration_step: int):
        log_strs = [f"iteration: {iteration_step} / {self.global_step}"]
        log_strs += [
            self.add_scalar(log_item.tag, log_item.scalar_value, log_item.log_format) for log_item in log_entries
        ]
        log_str = " | ".join(log_strs)
        log_rank(log_str, logger=get_logger(__name__), level=logging.INFO)


def set_logger_verbosity_format(logging_level: str, parallel_context: ParallelContext):
    node_name = os.environ.get("SLURMD_NODENAME")
    expert_parallel_log = (
        f"|EXP={dist.get_rank(parallel_context.expert_pg)}" if parallel_context.expert_parallel_size > 1 else ""
    )
    formatter = Formatter(
        fmt=f"%(asctime)s [%(levelname)s|DP={dist.get_rank(parallel_context.dp_pg)}|PP={dist.get_rank(parallel_context.pp_pg)}|"
        f"TP={dist.get_rank(parallel_context.tp_pg)}{expert_parallel_log}{'|' + node_name if node_name else ''}]: %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    # TODO @thomasw21: `logging.log_levels` returns valid lg log levels
    log_level = log_levels[logging_level]

    # main root logger
    root_logger = get_logger()
    root_logger.setLevel(log_level)
    handler = NewLineStreamHandler(sys.stdout)
    handler.setLevel(log_level)
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Nanotron
    set_verbosity(log_level)
    set_formatter(formatter=formatter)


_configure_library_root_logger()
