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
from logging import (
    CRITICAL,
    DEBUG,
    ERROR,
    FATAL,
    INFO,
    NOTSET,
    WARNING,
    Formatter,
    Logger,
)
from typing import TYPE_CHECKING, List, Optional, Union

import torch
from torch import distributed as torch_dist

from nanotron import distributed as dist

if TYPE_CHECKING:
    from nanotron.config import LoggingArgs
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


class CategoryFilter(logging.Filter):
    """Filter to add category field to log records."""

    def filter(self, record):
        # Add category attribute if not present
        if not hasattr(record, "category"):
            record.category = ""
        elif record.category:
            # Format the category if it exists
            record.category = f"|{record.category}"
        return True


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
    category: Optional[str] = None,
    is_separator: bool = False,
    **kwargs,
):
    """Log only if the current process is the rank specified."""
    # Use default group is group is not provided
    if group is None:
        group = torch_dist.distributed_c10d._get_default_group()

    # Add category to the extra kwargs
    if category is not None:
        kwargs["extra"] = kwargs.get("extra", {})
        kwargs["extra"]["category"] = category

    # Add separator to the extra kwargs
    if is_separator:
        kwargs["extra"] = kwargs.get("extra", {})
        kwargs["extra"]["separator"] = True

    # rank is None means everyone logs
    if rank is None or dist.get_rank(group) == rank:
        if is_separator:
            logger.log(level, "=" * 50, **kwargs)
        logger.log(level, msg, **kwargs)
        if is_separator:
            logger.log(level, "=" * 50, **kwargs)


@lru_cache(maxsize=None)
def warn_once(
    msg: str, logger: Logger, group: Optional[dist.ProcessGroup] = None, rank: Optional[int] = None, **kwargs
):
    log_rank(msg=msg, logger=logger, level=logging.WARNING, group=group, rank=rank, **kwargs)


def human_format(num: float, billions: bool = False, divide_by_1024: bool = False) -> str:
    if abs(num) < 1:
        return "{:.3g}".format(num)
    SIZES = ["", "K", "M", "B", "T", "P", "E"]
    num = float("{:.3g}".format(num))
    magnitude = 0
    i = 0
    while abs(num) >= 1000 and i < len(SIZES) - 1:
        magnitude += 1
        num /= 1000.0 if not divide_by_1024 else 1024.0
        i += 1
    return "{}{}".format("{:f}".format(num).rstrip("0").rstrip("."), SIZES[magnitude])


def log_memory(logger: logging.Logger, msg: str = ""):
    log_rank(
        f"{msg}\n"
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
    # 1. Conditional rank display - only show ranks if their size is > 1
    node_name = os.environ.get("SLURMD_NODENAME")
    ranks = []

    if parallel_context.expert_parallel_size > 1:
        ranks.append(f"EP={dist.get_rank(parallel_context.ep_pg)}")
    if parallel_context.context_parallel_size > 1:
        ranks.append(f"CP={dist.get_rank(parallel_context.cp_pg)}")
    if parallel_context.data_parallel_size > 1:
        ranks.append(f"DP={dist.get_rank(parallel_context.dp_pg)}")
    if parallel_context.pipeline_parallel_size > 1:
        ranks.append(f"PP={dist.get_rank(parallel_context.pp_pg)}")
    if parallel_context.tensor_parallel_size > 1:
        ranks.append(f"TP={dist.get_rank(parallel_context.tp_pg)}")

    if node_name:
        ranks.append(node_name)

    # Join all ranks with separator
    ranks_str = "|".join(ranks)
    ranks_display = f"|{ranks_str}" if ranks_str else ""

    # Use a custom formatter class that handles missing fields
    class SafeFormatter(Formatter):
        def format(self, record):
            # Ensure required attributes exist before formatting
            if not hasattr(record, "category"):
                record.category = ""
            elif record.category and not record.category.startswith("|"):
                record.category = f"|{record.category}"

            # Store original message for restoration later
            original_msg = record.msg

            # Apply styling based on record properties
            is_separator = getattr(record, "separator", False)
            if is_separator:
                record.msg = f"\033[1m{record.msg}\033[0m"  # Bold for separators

            # Choose color prefix/suffix based on log level
            if record.levelno == logging.WARNING:
                prefix = "\033[1;33m"  # Bold yellow for warnings
            elif record.levelno >= logging.ERROR:
                prefix = "\033[1;31m"  # Bold red for errors and critical
            elif record.levelno == logging.DEBUG:
                prefix = "\033[2;3;32m"  # Dim and italic green for debug
            else:
                prefix = "\033[2;3m"  # Dim and italic for other levels

            suffix = "\033[0m"

            # Save the original format
            original_fmt = self._style._fmt

            # Use a more consistent format with prefix/suffix applied only to the metadata portion
            self._style._fmt = f"{prefix}%(asctime)s [%(levelname)s%(category)s{ranks_display}]{suffix}: %(message)s"

            # Format the record
            result = super().format(record)

            # Restore the original values
            self._style._fmt = original_fmt
            record.msg = original_msg

            return result

    # Create formatter with the safe handling
    formatter = SafeFormatter(
        fmt=f"\033[2;3m%(asctime)s [%(levelname)s%(category)s{ranks_display}]\033[0m: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )
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


def set_ranks_logging_level(parallel_context: ParallelContext, logging_config: "LoggingArgs"):
    if dist.get_rank(parallel_context.world_pg) == 0:
        if logging_config.log_level is not None:
            set_logger_verbosity_format(logging_config.log_level, parallel_context=parallel_context)
    else:
        if logging_config.log_level_replica is not None:
            set_logger_verbosity_format(logging_config.log_level_replica, parallel_context=parallel_context)


def log_libraries_versions(logger: logging.Logger):
    import datasets
    import flash_attn
    import numpy
    import torch
    import transformers

    import nanotron

    if dist.get_rank() == 0:
        log_rank("Libraries versions:", logger=logger, level=logging.INFO, rank=0, is_separator=True)
        log_rank(f"nanotron version: {nanotron.__version__}", logger=logger, level=logging.INFO, rank=0)
        log_rank(f"torch version: {torch.__version__}", logger=logger, level=logging.INFO, rank=0)
        log_rank(f"transformers version: {transformers.__version__}", logger=logger, level=logging.INFO, rank=0)
        log_rank(f"datasets version: {datasets.__version__}", logger=logger, level=logging.INFO, rank=0)
        log_rank(f"flash-attn version: {flash_attn.__version__}", logger=logger, level=logging.INFO, rank=0)
        log_rank(f"numpy version: {numpy.__version__}", logger=logger, level=logging.INFO, rank=0)
        log_rank(
            f"\ntorch.utils.collect_env: {torch.utils.collect_env.main()}", logger=logger, level=logging.INFO, rank=0
        )


_configure_library_root_logger()
