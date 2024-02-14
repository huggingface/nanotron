from dataclasses import dataclass

from nanotron.fp8.dtypes import Dtypes


class FP8TensorRecipe:
    dtype: Dtypes
    window_size: int


@dataclass
class FP8LinearRecipe:
    pass


FP8LMRecipe = FP8LinearRecipe(
    
)
