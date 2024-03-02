import torch

from nanotron.fp8.constants import LS_INITIAL_SCALING_FACTOR, LS_INITIAL_SCALING_VALUE, LS_INTERVAL


class LossScaler:
    """Dynamic Loss Scaler for FP8 FP16 mixed precision training."""

    def __init__(
        self,
        scaling_value: torch.Tensor = LS_INITIAL_SCALING_VALUE,
        scaling_factor: torch.Tensor = LS_INITIAL_SCALING_FACTOR,
        interval: int = LS_INTERVAL,
    ):
        # NOTE: because the precision of these scaling factors
        # affect the precision of the gradients
        assert scaling_value.dtype == torch.float32
        assert scaling_factor.dtype == torch.float32

        self.scaling_value = scaling_value
        self.scaling_factor = scaling_factor
        self.interval = interval
        # NOTE: a variable that keep track the number of overflow
        # and underflow are detected during this interval
        self.overflow_counter: int = 0

        self.update()

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss * self.scaling_value

    def step(self, optim: torch.optim.Optimizer):
        detected_overflow = False
        for group in optim.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if is_overflow(p.grad):
                        detected_overflow = True
                        break

        if detected_overflow:
            # TODO(xrsrke): update scaling factor, and scaling value
            # TODO(xrsrke): add logging that we skip optimizer step when overflow
            # is detected
            if self.interval == 1:
                self.update()
        else:
            # NOTE: unscale gradients
            for group in optim.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        p.grad = p.grad / self.scaling_value

            # TODO(xrsrke): update scaling factor, and scaling value
            optim.step()

    def _update_new_scaling_factor(self):
        pass

    def update(self):
        self.scaling_value = self.scaling_value * self.scaling_factor
        self.overflow_counter = 0

    def _register_detect_overflow_underflow_hooks(self):
        pass

    # def _register_unscale_gradients_hooks(self, parameters: List[nn.Parameter], scaling_factor: torch.Tensor):
    #     for p in parameters:
    #         p.register_hook(lambda grad: grad / scaling_factor)


def is_overflow(tensor: torch.Tensor) -> bool:
    return torch.any(torch.isinf(tensor))


def is_underflow():
    pass
