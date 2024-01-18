import torch
from nanotron import logging
from nanotron.parallel.pipeline_parallel.p2p import P2P
from nanotron.parallel.pipeline_parallel.state import PipelineBatchState

logger = logging.get_logger(__name__)


class SendTensorToPipelineBuffer(torch.autograd.Function):
    """Make sending tensors differentiable. The difference is here we don't use `torch.distributed` primites, but store events that's we will pop whenever we need"""

    @staticmethod
    def forward(
        ctx,
        activation: torch.Tensor,
        to_rank: int,
        p2p: P2P,
        pipeline_state: PipelineBatchState,
    ):
        assert activation.requires_grad
        ctx.p2p = p2p
        ctx.to_rank = to_rank
        ctx.pipeline_state = pipeline_state

        # Send tensors
        pipeline_state.register_send_activation(activation, to_rank=to_rank, p2p=p2p)

        # HACK @thomasw21: This forces the trigger to backward
        return torch.tensor(1, dtype=torch.float, device="cpu", requires_grad=True)

    @staticmethod
    def backward(ctx, grad_tensor):
        p2p = ctx.p2p
        to_rank = ctx.to_rank
        pipeline_state = ctx.pipeline_state

        # send a gradient and store it in buffer
        pipeline_state.register_recv_grad(from_rank=to_rank, p2p=p2p)
        if len(pipeline_state.grads_buffer) == 0:
            pipeline_state.run_communication()

        grad_tensor = pipeline_state.grads_buffer.popleft()

        return grad_tensor, None, None, None


class SendTensorWithoutGradientToPipelineBuffer(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        dummy_input: torch.Tensor,
        activation: torch.Tensor,
        to_rank: int,
        p2p: P2P,
        pipeline_state: PipelineBatchState,
    ):
        assert dummy_input.requires_grad
        assert activation.requires_grad is False
        ctx.p2p = p2p
        ctx.to_rank = to_rank
        ctx.pipeline_state = pipeline_state

        # Send tensors
        pipeline_state.register_send_activation(activation, to_rank=to_rank, p2p=p2p)

        # HACK @thomasw21: This forces the trigger to backward
        return torch.tensor(1, dtype=torch.float, device="cpu", requires_grad=True)

    @staticmethod
    def backward(ctx, grad_tensor):
        pipeline_state = ctx.pipeline_state

        # send only the activations
        pipeline_state.run_communication(send_only_activation=True)

        return None, None, None, None, None


def send_to_pipeline_state_buffer(tensor: torch.Tensor, to_rank: int, p2p: P2P, pipeline_state: PipelineBatchState):
    # This is used in order to know where to backward from.
    if tensor.requires_grad:
        result = SendTensorToPipelineBuffer.apply(tensor, to_rank, p2p, pipeline_state)
    else:
        # Trick that backward mechanism to just send the tensor.
        dummy_input = torch.empty(1, dtype=torch.float, requires_grad=True, device="cpu")
        result = SendTensorWithoutGradientToPipelineBuffer.apply(dummy_input, tensor, to_rank, p2p, pipeline_state)

    pipeline_state.register_activation_requiring_backward(result)


class RecvTensorFromPipelineBuffer(torch.autograd.Function):
    """Make receiving tensors differentiable"""

    @staticmethod
    def forward(ctx, activation: torch.Tensor, from_rank: int, p2p: P2P, pipeline_state: PipelineBatchState):
        ctx.pipeline_state = pipeline_state
        ctx.p2p = p2p
        ctx.from_rank = from_rank

        return activation

    @staticmethod
    def backward(ctx, grad_tensor):
        pipeline_state = ctx.pipeline_state
        from_rank = ctx.from_rank
        p2p = ctx.p2p

        # Send tensors
        pipeline_state.register_send_grad(grad_tensor, to_rank=from_rank, p2p=p2p)

        return None, None, None, None


def recv_from_pipeline_state_buffer(from_rank: int, p2p: P2P, pipeline_state: PipelineBatchState):
    pipeline_state.register_recv_activation(from_rank=from_rank, p2p=p2p)
    if len(pipeline_state.activations_buffer) == 0:
        pipeline_state.run_communication()
    activation = pipeline_state.activations_buffer.popleft()
    return RecvTensorFromPipelineBuffer.apply(activation, from_rank, p2p, pipeline_state)
