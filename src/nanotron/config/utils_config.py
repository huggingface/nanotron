from dataclasses import fields
from enum import Enum, auto
from pathlib import Path

import torch

from nanotron.generation.sampler import SamplerType
from nanotron.parallel.pipeline_parallel.engine import (
    AllForwardAllBackwardPipelineEngine,
    OneForwardOneBackwardPipelineEngine,
    PipelineEngine,
)
from nanotron.parallel.tensor_parallel.nn import TensorParallelLinearMode


class RecomputeGranularity(Enum):
    SELECTIVE = auto()
    FULL = auto()


def serialize(data) -> dict:
    """Recursively serialize a nested dataclass to a dict - do some type conversions along the way"""
    if data is None:
        return None

    if not hasattr(data, "__dataclass_fields__"):
        return data

    result = {}
    for field in fields(data):
        value = getattr(data, field.name)
        if hasattr(value, "__dataclass_fields__"):
            result[field.name] = serialize(value)
        elif isinstance(value, Path):
            result[field.name] = str(value)
        elif isinstance(value, PipelineEngine):
            result[field.name] = cast_pipeline_engine_to_str(value)
        elif isinstance(value, TensorParallelLinearMode):
            result[field.name] = value.name
        elif isinstance(value, RecomputeGranularity):
            result[field.name] = value.name
        elif isinstance(value, SamplerType):
            result[field.name] = value.name
        elif isinstance(value, torch.dtype):
            result[field.name] = dtype_to_str[value]
        elif isinstance(value, (list, tuple)):
            result[field.name] = [serialize(v) for v in value]
        elif isinstance(value, dict) and not value:
            result[field.name] = None  # So we can serialize empty dicts without issue with `datasets` in particular
        else:
            result[field.name] = value

    return result


str_to_dtype = {
    "float32": torch.float32,
    "float64": torch.float64,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}

dtype_to_str = {
    torch.float32: "float32",
    torch.float64: "float64",
    torch.complex64: "complex64",
    torch.complex128: "complex128",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.uint8: "uint8",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.bool: "bool",
}


def cast_str_to_torch_dtype(str_dtype: str):
    if str_dtype in str_to_dtype:
        return str_to_dtype[str_dtype]
    else:
        raise ValueError(f"dtype should be a string selected in {str_to_dtype.keys()} and not {str_dtype}")


def cast_str_to_pipeline_engine(str_pp_engine: str) -> PipelineEngine:
    if str_pp_engine == "afab":
        return AllForwardAllBackwardPipelineEngine()
    elif str_pp_engine == "1f1b":
        return OneForwardOneBackwardPipelineEngine()
    else:
        raise ValueError(f"pp_engine should be a string selected in ['afab', '1f1b'] and not {str_pp_engine}")


def cast_pipeline_engine_to_str(pp_engine: PipelineEngine) -> str:
    if isinstance(pp_engine, AllForwardAllBackwardPipelineEngine):
        return "afab"
    elif isinstance(pp_engine, OneForwardOneBackwardPipelineEngine):
        return "1f1b"
    else:
        raise ValueError(
            f"pp_engine should be aan instance of AllForwardAllBackwardPipelineEngine or OneForwardOneBackwardPipelineEngine, not {type(pp_engine)}"
        )
