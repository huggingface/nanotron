from typing import List, Union

import torch
from torch import nn
from torch.optim import Optimizer

from nanotron.fp8.constants import LS_INITIAL_SCALING_FACTOR, LS_INITIAL_SCALING_VALUE, LS_INTERVAL
from nanotron.fp8.parameter import FP8Parameter
from nanotron.fp8.tensor import FP8Tensor, FP16Tensor


class LossScaler:
    """Dynamic Loss Scaler for FP8 & FP16 mixed precision training."""

    def __init__(
        self,
        scaling_value: torch.Tensor = LS_INITIAL_SCALING_VALUE,
        scaling_factor: torch.Tensor = LS_INITIAL_SCALING_FACTOR,
        interval: int = LS_INTERVAL,
    ):
        # NOTE: because the precision of these scaling factors
        # affect the precision of the gradients
        assert isinstance(scaling_value, torch.Tensor)
        assert isinstance(scaling_factor, torch.Tensor)
        assert scaling_value.dtype == torch.float32
        assert scaling_factor.dtype == torch.float32
        assert interval > 0

        self.scaling_value = scaling_value
        self.scaling_factor = scaling_factor
        self.interval = interval
        # NOTE: a variable that keep track the number of overflow
        # and underflow are detected during this interval
        self.overflow_counter = 0

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        # TODO(xrsrke): add autocast loss to float32 before scaling it
        # TODO(xrsrke): do inplace operation
        loss = loss.to(torch.float32) if loss.dtype != torch.float32 else loss
        return loss * self.scaling_value

    def step(self, optim: Optimizer, *args, **kwargs):
        # def sanity_overflow(optim):
        
            
        #     overflowed_params = []
        #     for group in optim.param_groups:
        #         for p in group["params"]:
        #             if p.grad is not None:
        #                 if is_overflow(p.grad):
        #                     overflowed_params.append(overflowed_params)
            
        #     for p in overflowed_params:
        #         param_to_name = {param: name for name, param in model.named_parameters()}
        #         print(param_to_name)
        
        
        detected_overflow = False
        for group in optim.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if is_overflow(p.grad):
                        detected_overflow = True
                        break

        if detected_overflow:
            # TODO(xrsrke): add logging that we skip optimizer step when overflow
            # is detected
            # TODO(xrsrke): remove this after debugging
            # raise RuntimeError("Detected overflow")
            print("Detected overflow, skipping optimizer step")
            self.update()
        else:
            # NOTE: unscale gradients
            for group in optim.param_groups:
                for p in group["params"]:
                    self._unscale_param_(p)

            optim.step(*args, **kwargs)

    def unscale_(self, params: List[Union[nn.Parameter, FP8Parameter]]):
        """Unscale the gradients of parameters in place."""
        for p in params:
            self._unscale_param_(p)

    def _unscale_param_(self, p: Union[nn.Parameter, FP8Parameter]):
        if p.requires_grad and p.grad is not None:
            if isinstance(p.grad, FP8Tensor) or isinstance(p.grad, FP16Tensor):
                p.grad.mul_(1 / self.scaling_value)
            else:
                p.grad.div_(self.scaling_value)

    def update(self):
        # TODO(xrsrke): remove this
        self.overflow_counter += 1

        if self.overflow_counter == self.interval:
            self.scaling_value = self.scaling_value * self.scaling_factor
            self.overflow_counter = 0


def is_overflow(tensor: torch.Tensor) -> bool:
    if torch.isinf(tensor).any() or torch.isnan(tensor).any():
        return True
    else:
        return False


def is_underflow():
    pass
