import math
from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from torch.optim import Optimizer

from nanotron.fp8.constants import FP8LM_RECIPE
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.parameter import FP8Parameter
from nanotron.fp8.tensor import FP8Tensor, convert_tensor_from_fp8, convert_tensor_to_fp8
from nanotron.fp8.utils import get_tensor_fp8_metadata
from nanotron.fp8.meta import FP8Meta


class Adam(Optimizer):
    r"""Implements Adam algorithm.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                
                print(f"[Ref Adam] original grad: {grad[:2, :2]} \n")
                
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
                
                print(f"[Ref Adam] original exp_avg: exp_avg.data={exp_avg.data[:2, :2]}, exp_avg.dtype={exp_avg.dtype} \n")
                print(f"[Ref Adam] original exp_avg_sq: exp_avg_sq.data={exp_avg_sq.data[:2, :2]}, exp_avg_sq.dtype={exp_avg_sq.dtype} \n")
                print(f"[Ref Adam] beta1: {beta1}, beta2: {beta2}")

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                print(f"[Ref Adam]: bias_correction1: {bias_correction1}, bias_correction2: {bias_correction2}")

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)
                    print(f"[Ref Adam] grad after weight decay: {grad[:2, :2]} \n")

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                
                print(f"[Ref Adam] after mul and add: exp_avg: {exp_avg[:2, :2]} \n")
                print(f"[Ref Adam] after mul and add: exp_avg_sq: {exp_avg_sq[:2, :2]} \n")
                
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                    print(f"[Ref Adam] exp_avg_sq.sqrt(): {exp_avg_sq.sqrt()[:2, :2]} \n")
                    print(f"[Ref Adam] math.sqrt(bias_correction2)): {math.sqrt(bias_correction2)} \n")
                    print(f"[Ref Adam] group['eps']: {group['eps']} \n")

                step_size = group['lr'] / bias_correction1
                print(f"[Ref Adam] step_size: {step_size} \n")
                print(f"[Ref Adam] exp_avg: {exp_avg[:2, :2]} \n")
                print(f"[Ref Adam] denom: {denom[:2, :2]} \n")

                p.data.addcdiv_(-step_size, exp_avg, denom)
                
                print(f"[Ref Adam] updated p: {p.data[:2, :2]} \n")
                
                break

        return loss


class FP8Adam(Optimizer):
    """
    FP8 Adam optimizer.
    Credits to the original non-FP8 Adam implementation: https://github.com/201419/Optimizer-PyTorch/blob/ce5c0dc96dca0689af2e0a3f0b0bb3821c2a31b0/adam.py#L6
    """

    def __init__(
        self,
        params: List[FP8Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
    ):
        # TODO(xrsrke): add this back, after fp8 working
        # assert [isinstance(p, FP8Parameter) for p in params], "All parameters should be FP8Parameter"

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "amsgrad": amsgrad}

        super().__init__(params, defaults)

        # TODO(xrsrke): make FP8Adam take a FP8Recipe
        # then retrieve the exp_avg_dtype from the recipe
        self.exp_avg_dtype = FP8LM_RECIPE.optim.exp_avg_dtype
        self.exp_avg_sq_dtype = FP8LM_RECIPE.optim.exp_avg_sq_dtype

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    def _init_optim_states(self, state: Dict[str, Any], p: nn.Parameter, amsgrad: bool) -> None:
        FP8_DTYPES = [DTypes.FP8E4M3, DTypes.FP8E5M2]

        # TODO(xrsrke): only cast to FP8Tensor if the dtype is FP8
        # state["exp_avg"] = FP8Tensor(torch.zeros(p.data.shape, memory_format=torch.preserve_format), dtype=self.exp_avg_dtype)

        # Exponential moving average of gradient values
        # TODO(xrsrke): maybe initialize a lower precision then cast to FP8Tensor
        # because zeros fp16 = zeros fp32?
        exp_avg = torch.zeros(p.data.shape, dtype=torch.float32, device="cuda")
        if self.exp_avg_dtype in FP8_DTYPES:
            exp_avg = FP8Tensor(exp_avg, dtype=self.exp_avg_dtype)

        # Exponential moving average of squared gradient values
        # TODO(xrsrke): don't fixed the dtype to fp16
        exp_avg_sq = torch.zeros(p.data.shape, dtype=torch.float32, device="cuda")
        if self.exp_avg_sq_dtype in FP8_DTYPES:
            exp_avg_sq = FP8Tensor(exp_avg_sq, dtype=self.exp_avg_dtype)

        state["step"] = 0
        state["exp_avg"] = exp_avg
        state["exp_avg_sq"] = exp_avg_sq

        if amsgrad:
            # Maintains max of all exp. moving avg. of sq. grad. values
            state["max_exp_avg_sq"] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                amsgrad = group["amsgrad"]
                if len(state) == 0:
                    self._init_optim_states(state, p, amsgrad)

                grad = p.grad.data
                print(f"[FP8Adam] original grad: {grad[:2, :2]}")
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]

                beta1, beta2 = group["betas"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                
                print(f"[FP8Adam] beta1: {beta1}, beta2: {beta2}")
                print(f"[FP8Adam]: bias_correction1: {bias_correction1}, bias_correction2: {bias_correction2}")

                if group["weight_decay"] != 0:
                    grad = grad.add(group["weight_decay"], p.data)
                    print(f"[FP8Adam] grad after weight decay: {grad[:2, :2]} \n")
                
                # Decay the first and second moment running average coefficient
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                
                print(f"[FP8Adam] original fp8 exp_avg: exp_avg.data={exp_avg.data[:2, :2]}, exp_avg.fp8_meta={exp_avg.fp8_meta} \n")

                # TODO(xrsrke): can we do all calculations in fp8?
                exp_avg_fp32 = convert_tensor_from_fp8(exp_avg, exp_avg.fp8_meta, torch.float32)
                print(f"[FP8Adam] exp_avg_fp32: {exp_avg_fp32[:2, :2]} \n")
                print(f"[FP8Adam] exp_avg_sq: {exp_avg_sq[:2, :2]} \n")

                exp_avg_fp32.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                
                print(f"[FP8Adam] after mul and add: exp_avg_fp32: {exp_avg_fp32[:2, :2]} \n")
                print(f"[FP8Adam] after mul and add: exp_avg_sq: {exp_avg_sq[:2, :2]} \n")

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])
                    print(f"[FP8Adam] denom: {denom[:2, :2]} \n")

                step_size = group["lr"] / bias_correction1
                print(f"[FP8Adam] step_size: {step_size} \n")

                # TODO(xrsrke): update optimizer states asyncronously
                # in a separate cuda streams
                exp_avg_fp32_meta = get_tensor_fp8_metadata(exp_avg_fp32, exp_avg.fp8_meta.dtype)
                updated_exp_avg_fp8 = convert_tensor_to_fp8(exp_avg_fp32, exp_avg_fp32_meta)
                
                print(f"[FP8Adam] updated_exp_avg_fp8: updated_exp_avg_fp8.data={updated_exp_avg_fp8.data[:2, :2]}, exp_avg_fp32_meta={exp_avg_fp32_meta} \n")
                
                exp_avg.copy_(updated_exp_avg_fp8)

                # TODO(xrsrke): can we do all calculations in fp8?
                p_fp32 = convert_tensor_from_fp8(p.data, p.fp8_meta, torch.float32)
                p_fp32.addcdiv_(-step_size, exp_avg_fp32, denom)
                print(f"[FP8Adam] updated p_fp32: {p_fp32[:2, :2]} \n")

                p_fp32_meta = get_tensor_fp8_metadata(p_fp32, dtype=p.data.fp8_meta.dtype)
                updated_p_fp8 = convert_tensor_to_fp8(p_fp32, p_fp32_meta)
                
                print(f"[FP8Adam] updated_p_fp8: updated_p_fp8.data={updated_p_fp8.data[:2, :2]}, p_fp32_meta={p_fp32_meta} \n")
                
                p.data.copy_(updated_p_fp8)
                
                break

    # TODO(xrsrke): refactor using strategy pattern
    def _update_scaling_factors(self):
        for p in self.param_groups:
            for param in p["params"]:
                if param.grad is None:
                    continue

                assert 1 == 1


def convert_tensor_to_fp16(tensor: torch.Tensor) -> torch.Tensor:
    from nanotron.fp8.constants import INITIAL_SCALING_FACTOR
    from nanotron.fp8.meta import FP8Meta
    from nanotron.fp8.tensor import update_scaling_factor

    dtype = DTypes.KFLOAT16

    amax = tensor.abs().max().clone()
    scale = update_scaling_factor(amax, torch.tensor(INITIAL_SCALING_FACTOR, dtype=torch.float32), dtype)
    meta = FP8Meta(amax, scale, dtype)

    # te_dtype = convert_torch_dtype_to_te_dtype(meta.dtype)
    # TODO(xrsrke): after casting to fp8, update the scaling factor
    # TODO(xrsrke): it's weird that TE only take inverse_scale equal to 1
    # inverse_scale = torch.tensor(1.0, device=tensor.device, dtype=torch.float32)
    return (tensor * meta.scale).to(torch.float16), meta


def convert_tensor_from_fp16(tensor: torch.Tensor, meta: FP8Meta, dtype: torch.dtype) -> torch.Tensor:
    # from nanotron.fp8.tensor import convert_torch_dtype_to_te_dtype, update_scaling_factor
    # from nanotron.fp8.constants import INITIAL_SCALING_FACTOR
    # from nanotron.fp8.meta import FP8Meta

    # dtype = DTypes.KFLOAT16

    # amax = tensor.abs().max().clone()
    # scale = update_scaling_factor(amax, torch.tensor(INITIAL_SCALING_FACTOR, dtype=torch.float32), dtype)
    # meta = FP8Meta(amax, scale, dtype)

    # te_dtype = convert_torch_dtype_to_te_dtype(meta.dtype)
    # TODO(xrsrke): after casting to fp8, update the scaling factor
    # TODO(xrsrke): it's weird that TE only take inverse_scale equal to 1
    # inverse_scale = torch.tensor(1.0, device=tensor.device, dtype=torch.float32)
    return (tensor * meta.inverse_scale).to(dtype)
