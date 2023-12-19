from dataclasses import asdict, is_dataclass

from nanotron.core.import_utils import _can_import_from_module, _is_package_available
from nanotron.logger.dataclass import LogItem
from nanotron.logger.logger_writer import LoggerWriter

__all__ = ["LogItem", "LoggerWriter"]

tensorboardx_available = _is_package_available("tensorboardX")

if tensorboardx_available:
    from tensorboardX import SummaryWriter

    from nanotron.logger.tensorboard_logger import BatchSummaryWriter

    __all__ = __all__ + ["BatchSummaryWriter", "SummaryWriter"]

huggingface_hub_available = _is_package_available("huggingface_hub")
hf_tensorboard_logger_available = _can_import_from_module("huggingface_hub", "HFSummaryWriter")

if huggingface_hub_available and hf_tensorboard_logger_available:
    from nanotron.logger.hub_tensorboard_logger import HubSummaryWriter

    __all__ = __all__ + ["HubSummaryWriter"]

def flatten_dict(nested: dict, sep="/") -> dict:
    """Flatten dictionary, list, tuple and concatenate nested keys with separator."""

    def rec(nest: dict, prefix: str, into: dict):
        for k, v in sorted(nest.items()):
            # if sep in k:
            #     raise ValueError(f"separator '{sep}' not allowed to be in key '{k}'")
            if isinstance(v, dict):
                rec(v, prefix + k + sep, into)
            elif isinstance(v, (list, tuple)):
                for i, vv in enumerate(v):
                    if isinstance(vv, dict):
                        rec(vv, prefix + k + sep + str(i) + sep, into)
                    else:
                        vv = (
                            vv.replace("|", "_").replace("\n", "_") if isinstance(vv, str) else vv
                        )  # Need this for markdown
                        into[prefix + k + sep + str(i)] = vv
            else:
                v = v.replace("|", "_").replace("\n", "_") if isinstance(v, str) else v  # Need this for markdown
                into[prefix + k] = v

    flat = {}
    rec(nested, "", flat)
    return flat

def obj_to_markdown(obj, convert_s3_links: bool = True) -> str:
    """Convert a (potentially nested) dataclass object or a dict in a readable markdown string for logging"""
    if is_dataclass(obj):
        obj = asdict(obj)
    config_dict = flatten_dict(obj)
    config_markdown = "| Key | Value |\n| --- | --- |\n"
    for key, value in config_dict.items():
        if convert_s3_links and "s3://" in str(value):
            s3_bucket, s3_prefix = str(value).replace("s3://", "").split("/", maxsplit=1)
            if not s3_prefix.endswith("/"):
                s3_prefix += "/"
            link_str = f"https://s3.console.aws.amazon.com/s3/buckets/{s3_bucket}?prefix={s3_prefix}"
            value = f'<a href="{link_str}" target="_blank"> {value} </a>'
        config_markdown += f"| {key} | {value} |\n"
    return config_markdown
