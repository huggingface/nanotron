# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Optional, Tuple

import torch
from torch import nn

from nanotron import distributed as dist
from nanotron.distributed import get_global_rank
from nanotron.parallel.parameters import NanotronParameter
from nanotron.parallel.sharded_parameters import (
    SplitConfig,
    create_sharded_parameter_from_config,
    mark_all_parameters_in_module_as_sharded,
)
from nanotron.parallel.tensor_parallel.distributed_differentiable_primitives import (
    differentiable_all_gather,
    differentiable_all_reduce_sum,
    differentiable_identity,
    differentiable_reduce_scatter_sum,
)
from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
from nanotron.parallel.tensor_parallel.functional import (
    column_linear,
    row_linear,
)
from nanotron.parallel.tied_parameters import create_tied_parameter


class TensorParallelColumnLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        pg: dist.ProcessGroup,
        mode: TensorParallelLinearMode,
        bias=True,
        device=None,
        dtype=None,
        async_communication: bool = False,
        contiguous_chunks: Optional[Tuple[int, ...]] = None,
    ):
        self.pg = pg
        self.world_size = pg.size()

        assert out_features % self.world_size == 0

        self.in_features = in_features
        self.out_features = out_features // self.world_size

        super().__init__(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        self.mode = mode
        self.async_communication = async_communication

        if contiguous_chunks is not None:
            assert (
                sum(contiguous_chunks) == out_features
            ), f"Sum of contiguous chunks ({sum(contiguous_chunks)}) must equal to out_features ({out_features})"
        split_config = SplitConfig(split_dim=0, contiguous_chunks=contiguous_chunks)

        mark_all_parameters_in_module_as_sharded(
            self,
            pg=self.pg,
            split_config=split_config,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return column_linear(
            input=x,
            weight=self.weight,
            bias=self.bias,
            group=self.pg,
            tp_mode=self.mode,
            async_communication=self.async_communication,
        )

    def extra_repr(self) -> str:
        return f"tp_rank={dist.get_rank(self.pg)}, {super().extra_repr()}, unsharded_out_features={self.out_features * self.world_size}"


class TensorParallelRowLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        pg: dist.ProcessGroup,
        mode: TensorParallelLinearMode,
        bias=True,
        device=None,
        dtype=None,
        async_communication: bool = False,
        contiguous_chunks: Optional[Tuple[int, ...]] = None,
    ):
        self.pg = pg
        self.world_size = pg.size()

        assert in_features % self.world_size == 0

        self.in_features = in_features // self.world_size
        self.out_features = out_features

        # No need to shard the bias term, only rank 0 would have it
        bias = dist.get_rank(self.pg) == 0 and bias

        super().__init__(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.mode = mode
        self.async_communication = async_communication
        if self.mode is TensorParallelLinearMode.ALL_REDUCE and self.async_communication:
            raise ValueError("async_communication is not supported for ALL_REDUCE mode")

        if contiguous_chunks is not None:
            assert (
                sum(contiguous_chunks) == in_features
            ), f"Sum of contiguous chunks ({sum(contiguous_chunks)}) must equal to in_features ({in_features})"

        split_config = SplitConfig(split_dim=1, contiguous_chunks=contiguous_chunks)

        self._mark_all_parameters_in_module_as_sharded(split_config)

    def _mark_all_parameters_in_module_as_sharded(self, split_config: SplitConfig):
        for name, param in list(self.named_parameters()):
            if name == "bias":
                # `bias` only exists in rank 0 because it's not sharded
                new_param = NanotronParameter(tensor=param)
            else:
                new_param = create_sharded_parameter_from_config(
                    parameter=param,
                    pg=self.pg,
                    split_config=split_config,
                )
            setattr(self, name, new_param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return row_linear(
            input=x,
            weight=self.weight,
            bias=self.bias,
            group=self.pg,
            tp_mode=self.mode,
            async_communication=self.async_communication,
        )

    def extra_repr(self) -> str:
        return f"tp_rank={dist.get_rank(self.pg)}, {super().extra_repr()}, unsharded_in_features={self.in_features * self.world_size}"


class TiedLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        pg: dist.ProcessGroup,
        mode: TensorParallelLinearMode,
        bias=True,
        device=None,
        dtype=None,
    ):
        self.pg = pg
        self.world_size = pg.size()
        self.mode = mode

        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        self._mark_all_parameters_in_module_as_tied()

    def _mark_all_parameters_in_module_as_tied(self):
        for name, param in list(self.named_parameters()):
            new_param = create_tied_parameter(
                parameter=param,
                name=name,
                global_ranks=tuple(sorted((get_global_rank(self.pg, i) for i in range(self.pg.size())))),
                reduce_op=None if self.mode is TensorParallelLinearMode.ALL_REDUCE else dist.ReduceOp.SUM,
                root_module=self,
            )
            setattr(self, name, new_param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        if self.mode is TensorParallelLinearMode.ALL_REDUCE:
            y = differentiable_identity(y, group=self.pg)
        elif self.mode is TensorParallelLinearMode.REDUCE_SCATTER:
            y = differentiable_all_gather(y, group=self.pg)
        else:
            raise ValueError(f"Got unexpected mode: {self.mode}.")

        return y


class TensorParallelEmbedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        pg: dist.ProcessGroup,
        mode: TensorParallelLinearMode,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
        _weight=None,
        device=None,
        dtype=None,
        contiguous_chunks: Optional[Tuple[int, ...]] = None,
    ):
        self.pg = pg
        self.rank = dist.get_rank(self.pg)
        self.world_size = pg.size()

        self.original_num_embeddings = num_embeddings

        # TODO @thomasw21: Fix and remove that constraint. Typically there's no reason to have such a constraint.
        assert num_embeddings % self.world_size == 0
        block_size = num_embeddings // self.world_size
        # inputs in `[min_id, max_id[` are handled by `self` to get embeddings
        self.min_id = self.rank * block_size
        self.max_id = (self.rank + 1) * block_size

        super().__init__(
            block_size,
            embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            _weight=_weight,
            device=device,
            dtype=dtype,
        )

        self.mode = mode

        if contiguous_chunks is not None:
            assert (
                sum(contiguous_chunks) == num_embeddings
            ), f"Sum of contiguous chunks ({sum(contiguous_chunks)}) must equal to num_embeddings ({num_embeddings})"

        split_config = SplitConfig(split_dim=0, contiguous_chunks=contiguous_chunks)

        mark_all_parameters_in_module_as_sharded(self, pg=self.pg, split_config=split_config)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.pg.size() > 1:
            # `0` if input is in the correct interval, else `1`
            input_mask = torch.logical_or(self.min_id > input_ids, input_ids >= self.max_id)
            # translate for [0, self.max_id - self.min_id[
            masked_input = input_ids.clone() - self.min_id
            # default all out of bounds values to `0`
            masked_input[input_mask] = 0
        else:
            masked_input = input_ids
        out = super().forward(masked_input)

        if self.pg.size() > 1:
            out = out * (~input_mask[..., None])

        if self.mode is TensorParallelLinearMode.ALL_REDUCE:
            out = differentiable_all_reduce_sum(out, group=self.pg)
        elif self.mode is TensorParallelLinearMode.REDUCE_SCATTER:
            out = differentiable_reduce_scatter_sum(out, group=self.pg)
        else:
            raise ValueError(f"Got unexpected mode: {self.mode}.")

        return out

    def extra_repr(self) -> str:
        return f"tp_rank={dist.get_rank(self.pg)}, {super().extra_repr()}, unsharded_num_embeddings={self.original_num_embeddings}"
