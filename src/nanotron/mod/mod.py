from typing import Tuple, Union

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torchtyping import TensorType

from nanotron.parallel.pipeline_parallel.block import TensorPointer


class MixtureOfDepth(nn.Module):
    def __init__(self, capacity: int, d_model: int, block: nn.Module):
        super().__init__()
        self.router = Router(capacity, d_model)
        self.block = block

    # def forward(self, inputs: TensorType["batch_size", "seq_len", "d_model"]) -> TensorType["batch_size", "seq_len", "d_model"]:
    def forward(
        self,
        hidden_states: Union[TensorType["batch_size", "seq_len", "d_model"], TensorPointer],
        sequence_mask: Union[TensorType["batch_size", "seq_len"], TensorPointer],
    ) -> Tuple[
        Union[TensorType["batch_size", "seq_len", "d_model"], TensorPointer],
        Union[TensorType["batch_size", "seq_len"], TensorPointer],
    ]:
        hidden_states = rearrange(hidden_states, "seq_len batch_size d_model -> batch_size seq_len d_model")
        selected_idxs = self.router(hidden_states)
        assert selected_idxs.shape == (hidden_states.size(0), self.router.capacity)
        selected_hidden_states = hidden_states[torch.arange(hidden_states.size(0)).unsqueeze(1), selected_idxs]
        selected_sequence_mask = sequence_mask[torch.arange(sequence_mask.size(0)).unsqueeze(1), selected_idxs]

        selected_hidden_states = rearrange(
            selected_hidden_states, "batch_size seq_len d_model -> seq_len batch_size d_model"
        )
        outputs_of_selected_inputs = self.block(selected_hidden_states, selected_sequence_mask)
        # NOTE: now keep the representation of the selected inputs and replace the original inputs with the new ones
        hidden_states[torch.arange(hidden_states.size(0)).unsqueeze(1), selected_idxs] = rearrange(
            outputs_of_selected_inputs["hidden_states"], "seq_len batch_size d_model -> batch_size seq_len d_model"
        )
        hidden_states = rearrange(hidden_states, "batch_size seq_len d_model -> seq_len batch_size d_model")
        return {"hidden_states": hidden_states, "sequence_mask": sequence_mask}


class Router(nn.Module):
    def __init__(
        self,
        capacity: int,
        d_model: int,
        # tp_pg: dist.ProcessGroup,
        # parallel_config: Optional[ParallelismArgs]
    ):
        super().__init__()
        self.capacity = capacity
        self.gate = nn.Linear(d_model, 1)

        # TODO(xrsrke): deduplicate this
        # tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE
        # tp_linear_async_communication = (
        #     parallel_config.tp_linear_async_communication if parallel_config is not None else False
        # )

        # self.gate = TensorParallelRowLinear(
        #     d_model,
        #     1,
        #     pg=tp_pg,
        #     mode=TensorParallelLinearMode.REDUCE_SCATTER,
        #     bias=False,
        #     async_communication=True,
        #     # contiguous_chunks=gate_up_contiguous_chunks,
        # )

    def forward(self, inputs: TensorType["batch_size", "seq_len", "d_model"]) -> TensorType["batch_size", "seq_len"]:
        probs = F.softmax(self.gate(inputs), dim=1).view(-1, inputs.size(1))
        _, top_k_indices = torch.topk(probs, self.capacity)
        return top_k_indices
