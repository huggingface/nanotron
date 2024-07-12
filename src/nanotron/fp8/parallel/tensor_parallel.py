from nanotron.fp8.linear import FP8Linear
from nanotron.parallel.tensor_parallel.nn import TensorParallelColumnLinear


class TensorParallelColumnLinear(FP8Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
