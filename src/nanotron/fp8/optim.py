import math
from typing import Any, Dict, List, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
from torch.optim import Optimizer

from nanotron._utils.memory import delete_tensor_from_memory
from nanotron.fp8.constants import FP8_DTYPES, FP8LM_RECIPE
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.parameter import FP8Parameter
from nanotron.fp8.tensor import (
    FP8Tensor,
    FP16Tensor,
    convert_tensor_from_fp8,
    convert_tensor_from_fp16,
)
from nanotron.fp8.utils import compute_stas, is_overflow_underflow_nan
from nanotron.parallel.parameters import NanotronParameter


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

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "amsgrad": amsgrad}
        super(Adam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    # @snoop
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        loggings = {}
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                data = p.data
                assert isinstance(data, torch.Tensor)

                if len(state) == 0:
                    state["step"] = torch.tensor(0.0, dtype=data.dtype)
                    state["exp_avg"] = torch.zeros_like(data, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(data, memory_format=torch.preserve_format)

                loggings[p] = {}

                assert (p.grad is not None and p.data.grad is not None) is False
                grad = p.grad if p.grad is not None else p.data.grad
                assert isinstance(grad, torch.Tensor)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                # if group["weight_decay"] != 0:
                #     grad = grad.add(group["weight_decay"], data)

                # Decay the first and second moment running average coefficient
                # exp_avg.mul_(beta1).add_(1 - beta1, grad)
                # exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad.pow(2)

                step = state["step"]
                step += 1
                bias_correction1 = 1 - (beta1**step)
                bias_correction2 = 1 - (beta2**step)

                exp_avg = exp_avg / bias_correction1
                exp_avg_sq = exp_avg_sq / bias_correction2

                # denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])
                denom = exp_avg_sq.sqrt() + group["eps"]
                normalized_grad = exp_avg / denom

                lr = group["lr"]
                # p.data.addcdiv_(-step_size, exp_avg, denom)
                new_data = data - lr * normalized_grad
                new_data.requires_grad = True
                p.data = new_data

                assert p.data is new_data

                state["exp_avg"] = exp_avg
                state["exp_avg_sq"] = exp_avg_sq
                state["step"] = step

                loggings[p]["hp_grad"] = compute_stas(grad)
                loggings[p]["hp_p"] = compute_stas(p)
                loggings[p]["group:lr"] = {"value": lr}
                loggings[p]["group:eps"] = {"value": group["eps"]}
                loggings[p]["hp_exp_avg"] = compute_stas(exp_avg)
                loggings[p]["hp_exp_avg_sq"] = compute_stas(exp_avg_sq)
                loggings[p]["group:beta1"] = {"value": beta1}
                loggings[p]["group:beta2"] = {"value": beta2}

                loggings[p]["bias_correction1"] = {"value": bias_correction1}
                loggings[p]["bias_correction2"] = {"value": bias_correction2}
                loggings[p]["denom"] = compute_stas(denom)
                loggings[p]["normalized_grad"] = compute_stas(normalized_grad)

        self.loggings = loggings

        return loss

    def zero_grad(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad = None

                if p.data.grad is not None:
                    p.data.grad = None

                assert p.grad is None
                assert p.data.grad is None


class FP8Adam(Optimizer):
    """
    FP8 Adam optimizer.
    Credits to the original non-FP8 Adam implementation: https://github.com/201419/Optimizer-PyTorch/blob/ce5c0dc96dca0689af2e0a3f0b0bb3821c2a31b0/adam.py#L6
    """

    def __init__(
        self,
        params: List[nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        # amsgrad: bool = False,
        accum_dtype: DTypes = FP8LM_RECIPE.optim.accum_dtype,
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

        assert accum_dtype in DTypes, "Please provide an accumulation precision format"

        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "amsgrad": False}

        super().__init__(params, defaults)

        # TODO(xrsrke): make FP8Adam take a FP8Recipe
        # then retrieve the exp_avg_dtype from the recipe
        self.exp_avg_dtype = FP8LM_RECIPE.optim.exp_avg_dtype
        self.exp_avg_sq_dtype = FP8LM_RECIPE.optim.exp_avg_sq_dtype
        self.accum_dtype = accum_dtype

        self.master_weights: List[FP16Tensor] = []
        # NOTE: torch.Tensor is bias
        self.fp8_weights: List[Union[FP8Parameter, torch.Tensor]] = []
        # NOTE: use to map fp8 param to master weights
        self.mappping_fp8_to_master_weight: Dict[FP8Tensor, Union[FP16Tensor, torch.Tensor]] = {}

        for group in self.param_groups:
            for p in group["params"]:
                assert p.dtype == p.data.dtype

                # if p.requires_grad is None:
                #     continue

                # if p.data.dtype == torch.float16:
                #     continue

                # NOTE: if a tensor.ndim == 1, it's a bias
                # raw_data = p.data if p.ndim == 1 else p.orig_data
                # TODO(xrsrke): remove orig_data after FP8 working
                # raw_data = p.orig_data

                if isinstance(p, NanotronParameter):
                    raw_data = p.data.orig_data if hasattr(p.data, "orig_data") else p.data
                else:
                    raw_data = p.orig_data if hasattr(p, "orig_data") else p.data

                # if raw_data.dtype == torch.float16:
                #     # NOTE: only do mixed precision for parameters that quantize to FP8
                #     continue

                assert raw_data.dtype in [torch.float32]

                # if isinstance(p, NanotronParameter):
                #     assert 1 == 1
                # if "mlp.down_proj" in constants.PARAM_ID_TO_PARAM_NAMES[id(p)]:
                #     assert 1 == 1

                # TODO(xrsrke): retrieve the dtype for master_weights from the recipe
                fp16_p = FP16Tensor(raw_data, dtype=DTypes.KFLOAT16)

                # self.mappping_fp8_to_master_weight[p.data_ptr()] = fp16_p
                self.mappping_fp8_to_master_weight[p] = fp16_p
                self.master_weights.append(fp16_p)
                self.fp8_weights.append(p.data)

                delete_tensor_from_memory(raw_data)

                p.orig_data = None
                if hasattr(p.data, "orig_data"):
                    p.data.orig_data = None

                # if p.data.dtype == torch.float16:
                #     assert 1 == 1

        assert len(self.master_weights) == len(self.fp8_weights)
        # TODO(xrsrke): auto free fp32 weights from memory

        self.loggings = []
        self._is_overflow = False

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    # def _set_logging(self):
    #     pass

    def _init_optim_states(
        self,
        state: Dict[str, Any],
        p: nn.Parameter,
        # amsgrad: bool
    ) -> None:
        # TODO(xrsrke): move this to constants.py

        # TODO(xrsrke): only cast to FP8Tensor if the dtype is FP8
        # state["exp_avg"] = FP8Tensor(torch.zeros(p.data.shape, memory_format=torch.preserve_format), dtype=self.exp_avg_dtype)

        # Exponential moving average of gradient values
        # TODO(xrsrke): maybe initialize a lower precision then cast to FP8Tensor
        # because zeros fp16 = zeros fp32?
        # exp_avg = torch.zeros(p.data.shape, dtype=torch.float32, device="cuda")
        # exp_avg = FP8Tensor(exp_avg, dtype=self.exp_avg_dtype)

        # Exponential moving average of squared gradient values
        # TODO(xrsrke): don't fixed the dtype to fp16
        # exp_avg_sq = torch.zeros(p.data.shape, dtype=torch.float32, device="cuda")
        # exp_avg_sq = FP16Tensor(exp_avg_sq, dtype=DTypes.KFLOAT16)

        exp_avg = torch.zeros(p.data.shape, dtype=torch.float32, device="cuda")
        exp_avg_sq = torch.zeros(p.data.shape, dtype=torch.float32, device="cuda")

        state["step"] = torch.tensor(0.0, dtype=torch.float32, device="cuda")
        state["exp_avg"] = exp_avg
        state["exp_avg_sq"] = exp_avg_sq

        # if amsgrad:
        #     # Maintains max of all exp. moving avg. of sq. grad. values
        #     state["max_exp_avg_sq"] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

    # @torchsnooper.snoop()
    # @snoop
    @torch.no_grad()
    def step(self):
        # NOTE: sanity check the entire params has at least one grad
        # assert any(p.grad is not None for group in self.param_groups for p in group["params"])

        # TODO(xrsrke): remove this after debugging
        num_param_has_grads = 0
        for g in self.param_groups:
            for p in g["params"]:
                if p.data.__class__ == torch.Tensor:
                    if p.grad is not None:
                        num_param_has_grads += 1
                elif p.data.__class__ == FP8Tensor:
                    if hasattr(p.data, "_temp_grad") and p.data._temp_grad is not None:
                        num_param_has_grads += 1

        assert num_param_has_grads > 0

        self._is_overflow = False
        loggings = {}

        for i, group in enumerate(self.param_groups):
            for p in group["params"]:
                loggings[p] = {}

                state = self.state[p]
                if len(state) == 0:
                    self._init_optim_states(state, p)

                IS_FP8 = p.data.__class__ == FP8Tensor

                if IS_FP8:
                    assert p in self.mappping_fp8_to_master_weight, "FP8Tensor should have a master weight"
                    fp16_data = self.mappping_fp8_to_master_weight[p]
                    fp32_data = convert_tensor_from_fp16(fp16_data, torch.float32)
                    grad = p.data._temp_grad
                    fp32_grad = convert_tensor_from_fp8(grad, grad.fp8_meta, torch.float32)
                else:
                    fp16_data = p.data
                    fp32_data = fp16_data.to(torch.float32)
                    # NOTE: the bias of FP8 parameter saves its gradient in p.data.grad
                    # and the weight, and bias of non-FP8 parameter saves its gradient in p.grad
                    try:
                        assert (p.data.grad is None and p.grad is None) is False
                    except:
                        assert 1 == 1

                    grad = p.data.grad if p.data.grad is not None else p.grad
                    fp32_grad = grad.to(torch.float32)

                if p.__class__ == NanotronParameter:
                    if IS_FP8:
                        assert p.data.dtype in FP8_DTYPES
                    else:
                        assert p.data.dtype == torch.float16
                else:
                    assert (isinstance(p, FP8Parameter) and p.dtype in FP8_DTYPES) or (
                        isinstance(p, torch.Tensor) and p.dtype == torch.float16
                    ), f"type(p)={type(p)}, p.dtype={p.dtype}"

                assert (isinstance(grad, FP8Tensor) and grad.dtype in FP8_DTYPES) or (
                    isinstance(grad, torch.Tensor) and grad.dtype == torch.float16
                )

                loggings[p]["hp_grad"] = compute_stas(fp32_grad)

                if is_overflow_underflow_nan(fp32_grad):
                    self._is_overflow = True
                    raise ValueError("Overflow, underflow, or NaN detected in the gradients")

                if fp32_grad.is_sparse:
                    raise RuntimeError("FP8Adam does not support sparse gradients!")

                beta1, beta2 = group["betas"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                loggings[p]["group:beta1"] = {"value": beta1}
                loggings[p]["group:beta2"] = {"value": beta2}
                loggings[p]["bias_correction1"] = {"value": bias_correction1}
                loggings[p]["bias_correction2"] = {"value": bias_correction2}

                # TODO(xrsrke): can we do all calculations in fp8?
                # NOTE: somehow the view of bias changed, but the storage is the same
                # so we can't do the mapping, so now we map data_ptr to data_ptr
                # TODO(xrsrke): ideally, we should map tensor to tensor
                # it's easier to debug (u know which tensor is which)
                # loggings[p]["lp_p"] = compute_stas(fp16_data)

                assert fp16_data.dtype == torch.float16

                loggings[p]["hp_p"] = compute_stas(fp32_data)

                assert fp32_data.dtype == torch.float32
                assert fp32_grad.dtype == torch.float32

                if group["weight_decay"] != 0:
                    fp32_grad = fp32_grad.add(group["weight_decay"], fp32_data)

                # Decay the first and second moment running average coefficient
                fp32_exp_avg, fp32_exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                assert fp32_exp_avg.dtype == torch.float32
                assert fp32_exp_avg_sq.dtype == torch.float32

                loggings[p]["hp_exp_avg"] = compute_stas(fp32_exp_avg)
                loggings[p]["hp_exp_avg_sq"] = compute_stas(fp32_exp_avg_sq)

                assert fp32_exp_avg.dtype == torch.float32
                assert fp32_exp_avg_sq.dtype == torch.float32

                fp32_exp_avg.mul_(beta1).add_(1 - beta1, fp32_grad)
                fp32_exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, fp32_grad, fp32_grad)

                denom = (fp32_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])
                loggings[p]["denom"] = compute_stas(denom)

                if i == 4 and dist.get_rank() == 0:
                    assert 1 == 1

                # if p.__class__ == NanotronParameter:
                #     # NOTE: "initial_lr" is the peak lr
                #     # "lr" is the current lr based on scheduler
                #     # lr = group["initial_lr"] if "initial_lr" in group else group["lr"]
                #     lr = group["lr"]
                # else:
                #     lr = group["lr"]

                lr = group["lr"]
                step_size = lr / bias_correction1

                loggings[p]["group:lr"] = {"value": lr}
                loggings[p]["group:eps"] = {"value": group["eps"]}
                loggings[p]["step_size"] = {"value": step_size}

                new_fp32 = fp32_data - step_size * (fp32_exp_avg / denom)

                # assert not torch.allclose(new_fp32, fp32_data)

                if IS_FP8:
                    self.mappping_fp8_to_master_weight[p] = FP16Tensor(new_fp32, dtype=DTypes.KFLOAT16)
                    p.data.set_data(new_fp32)
                    # assert torch.allclose(p.data, new_fp32)
                else:
                    new_fp16 = new_fp32.to(torch.float16)
                    new_fp16.requires_grad = True
                    p.data = new_fp16
                    delete_tensor_from_memory(new_fp16)
                    assert torch.allclose(p.data, new_fp16)

                delete_tensor_from_memory(new_fp32)
                delete_tensor_from_memory(denom)

        self.loggings = loggings

    def zero_grad(self):
        for group in self.param_groups:
            for p in group["params"]:
                # NOTE: take the assumption that nanotron requires all parameters to have gradients
                # if (p.data.__class__ == FP8Tensor or not hasattr(p.data, "_temp_grad")) or \
                #     (p.data.__class__ == FP8Tensor and hasattr(p.data, "_temp_grad") and p.data._temp_grad is None) or \
                #     (p.data.__class__ == torch.Tensor and p.grad is None):
                #     continue

                if p.data.__class__ == FP8Tensor:
                    if hasattr(p.data, "_temp_grad") and p.data._temp_grad is not None:
                        delete_tensor_from_memory(p.data._temp_grad)
                        p.data._temp_grad = None
                else:
                    if p.grad is not None:
                        delete_tensor_from_memory(p.grad)
                        p.grad = None

                    if p.data.grad is not None:
                        delete_tensor_from_memory(p.data.grad)
                        p.data.grad = None
