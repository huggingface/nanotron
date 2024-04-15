import math
from typing import Any, Dict, List, Tuple, Union

import torch
from torch import nn
from torch.optim import Optimizer

from nanotron.fp8.constants import FP8_DTYPES, FP8LM_RECIPE
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.parameter import FP8Parameter
from nanotron.fp8.tensor import (
    FP8Tensor,
    FP16Tensor,
    convert_tensor_from_fp8,
    convert_tensor_from_fp16,
)
from nanotron.fp8.utils import is_overflow_underflow_nan, compute_stas


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
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                if p.ndim != 1:
                    print(f"[Ref Adam] original grad: {grad[:2, :2]} \n")

                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")
                amsgrad = group["amsgrad"]

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]

                if p.ndim != 1:
                    print(
                        f"[Ref Adam] original exp_avg: exp_avg.data={exp_avg.data[:2, :2]}, exp_avg.dtype={exp_avg.dtype} \n"
                    )
                    print(
                        f"[Ref Adam] original exp_avg_sq: exp_avg_sq.data={exp_avg_sq.data[:2, :2]}, exp_avg_sq.dtype={exp_avg_sq.dtype} \n"
                    )
                    print(f"[Ref Adam] beta1: {beta1}, beta2: {beta2}")

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                if p.ndim != 1:
                    print(f"[Ref Adam]: bias_correction1: {bias_correction1}, bias_correction2: {bias_correction2}")

                if group["weight_decay"] != 0:
                    grad = grad.add(group["weight_decay"], p.data)
                    if p.ndim != 1:
                        print(f"[Ref Adam] grad after weight decay: {grad[:2, :2]} \n")

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                if p.ndim != 1:
                    print(f"[Ref Adam] after mul and add: exp_avg: {exp_avg[:2, :2]} \n")
                    print(f"[Ref Adam] after mul and add: exp_avg_sq: {exp_avg_sq[:2, :2]} \n")

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])

                    if p.ndim != 1:
                        print(f"[Ref Adam] exp_avg_sq.sqrt(): {exp_avg_sq.sqrt()[:2, :2]} \n")
                        print(f"[Ref Adam] math.sqrt(bias_correction2)): {math.sqrt(bias_correction2)} \n")
                        print(f"[Ref Adam] group['eps']: {group['eps']} \n")

                step_size = group["lr"] / bias_correction1

                if p.ndim != 1:
                    print(f"[Ref Adam] step_size: {step_size} \n")
                    print(f"[Ref Adam] exp_avg: {exp_avg[:2, :2]} \n")
                    print(f"[Ref Adam] denom: {denom[:2, :2]} \n")

                p.data.addcdiv_(-step_size, exp_avg, denom)

                if p.ndim != 1:
                    print(f"[Ref Adam] updated p: {p.data[:2, :2]} \n")

            #     break
            # break

        return loss


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
                if p.requires_grad is None:
                    continue

                # NOTE: if a tensor.ndim == 1, it's a bias
                # raw_data = p.data if p.ndim == 1 else p.orig_data
                # TODO(xrsrke): remove orig_data after FP8 working
                # raw_data = p.orig_data
                raw_data = p.orig_data if hasattr(p, "orig_data") else p.data
                assert raw_data.dtype == torch.float32

                # TODO(xrsrke): retrieve the dtype for master_weights from the recipe
                fp16_p = FP16Tensor(raw_data, dtype=DTypes.KFLOAT16)

                # self.mappping_fp8_to_master_weight[p.data_ptr()] = fp16_p
                self.mappping_fp8_to_master_weight[p] = fp16_p
                self.master_weights.append(fp16_p)
                self.fp8_weights.append(p.data)

        assert len(self.master_weights) == len(self.fp8_weights)
        # TODO(xrsrke): auto free fp32 weights from memory
        
        self.loggings = []

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
        exp_avg = torch.zeros(p.data.shape, dtype=torch.float32, device="cuda")
        exp_avg = FP8Tensor(exp_avg, dtype=self.exp_avg_dtype)

        # Exponential moving average of squared gradient values
        # TODO(xrsrke): don't fixed the dtype to fp16
        exp_avg_sq = torch.zeros(p.data.shape, dtype=torch.float32, device="cuda")
        exp_avg_sq = FP16Tensor(exp_avg_sq, dtype=DTypes.KFLOAT16)

        state["step"] = 0
        state["exp_avg"] = exp_avg
        state["exp_avg_sq"] = exp_avg_sq

        # if amsgrad:
        #     # Maintains max of all exp. moving avg. of sq. grad. values
        #     state["max_exp_avg_sq"] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

    # @torchsnooper.snoop()
    # @snoop
    def step(self):
        # NOTE: sanity check the entire params has at least one grad
        assert any(p.grad is not None for group in self.param_groups for p in group["params"])
        loggings = {}
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                loggings[p] = {}

                state = self.state[p]
                if len(state) == 0:
                    self._init_optim_states(state, p)

                grad = p.grad
                # loggings[p]["lp_grad"] = compute_stas(grad)

                # NOTE: sanity check
                assert (isinstance(p, FP8Parameter) and p.dtype in FP8_DTYPES) or (
                    isinstance(p, torch.Tensor) and p.dtype == torch.float16
                ), f"type(p)={type(p)}, p.dtype={p.dtype}"
                
                assert (isinstance(grad, FP8Tensor) and grad.dtype in FP8_DTYPES) or (
                    isinstance(grad, torch.Tensor) and grad.dtype == torch.float16
                )

                if p.ndim != 1:
                    print(f"[FP8Adam] original grad: {grad[:2, :2]} \n")

                fp32_grad = (
                    grad.to(torch.float32)
                    if p.ndim == 1
                    else convert_tensor_from_fp8(grad, grad.fp8_meta, torch.float32)
                )
                loggings[p]["hp_grad"] = compute_stas(fp32_grad)

                if is_overflow_underflow_nan(fp32_grad):
                    # print(f"Overflow, underflow, or NaN detected in the gradients. So skip the current step")
                    continue
                    # raise ValueError("Overflow, underflow, or NaN detected in the gradients")

                if fp32_grad.is_sparse:
                    raise RuntimeError("FP8Adam does not support sparse gradients!")

                if p.ndim != 1:
                    print(f"[FP8Adam] fp32_grad: {fp32_grad[:2, :2]} \n")

                beta1, beta2 = group["betas"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                if p.ndim != 1:
                    print(f"[FP8Adam] beta1: {beta1}, beta2: {beta2}")
                    print(f"[FP8Adam]: bias_correction1: {bias_correction1}, bias_correction2: {bias_correction2}")

                # TODO(xrsrke): can we do all calculations in fp8?
                # NOTE: somehow the view of bias changed, but the storage is the same
                # so we can't do the mapping, so now we map data_ptr to data_ptr
                # TODO(xrsrke): ideally, we should map tensor to tensor
                # it's easier to debug (u know which tensor is which)
                fp16_p = self.mappping_fp8_to_master_weight[p]
                # loggings[p]["lp_p"] = compute_stas(fp16_p)
                assert fp16_p.dtype == torch.float16
                fp32_p = convert_tensor_from_fp16(fp16_p, torch.float32)
                loggings[p]["hp_p"] = compute_stas(fp32_p)

                assert fp32_p.dtype == torch.float32
                assert fp32_grad.dtype == torch.float32

                if p.ndim != 1:
                    print(f"[FP8Adam] fp16_p: {fp16_p[:2, :2]} \n")
                    print(f"[FP8Adam] fp32_p: {fp32_p[:2, :2]} \n")

                if group["weight_decay"] != 0:
                    fp32_grad = fp32_grad.add(group["weight_decay"], fp32_p)
                    if p.ndim != 1:
                        print(f"FP8Adam] group['weight_decay']: {group['weight_decay']}")
                        print(f"[FP8Adam] grad after weight decay: {fp32_grad[:2, :2]} \n")

                # Decay the first and second moment running average coefficient
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                assert exp_avg.dtype == torch.uint8
                assert exp_avg_sq.dtype == torch.float16
                
                # loggings[p]["lp_exp_avg"] = compute_stas(exp_avg)
                # loggings[p]["lp_exp_avg_sq"] = compute_stas(exp_avg_sq)

                if p.ndim != 1:
                    print(
                        f"[FP8Adam] original fp8 exp_avg: exp_avg.data={exp_avg.data[:2, :2]}, exp_avg.fp8_meta={exp_avg.fp8_meta} \n"
                    )
                    print(
                        f"[FP8Adam] original fp16 exp_avg_sq: exp_avg_sq.data={exp_avg_sq.data[:2, :2]}, exp_avg_sq.dtype={exp_avg_sq.dtype} \n"
                    )

                # TODO(xrsrke): can we do all calculations in fp8?
                fp32_exp_avg = convert_tensor_from_fp8(exp_avg, exp_avg.fp8_meta, torch.float32)
                fp32_exp_avg_sq = convert_tensor_from_fp16(exp_avg_sq, torch.float32)
                
                loggings[p]["hp_exp_avg"] = compute_stas(fp32_exp_avg)
                loggings[p]["hp_exp_avg_sq"] = compute_stas(fp32_exp_avg_sq)

                assert fp32_exp_avg.dtype == torch.float32
                assert fp32_exp_avg_sq.dtype == torch.float32

                if p.ndim != 1:
                    print(f"[FP8Adam] fp32_exp_avg: {fp32_exp_avg[:2, :2]} \n")
                    print(f"[FP8Adam] fp32_exp_avg_sq: {fp32_exp_avg_sq[:2, :2]} \n")

                fp32_exp_avg.mul_(beta1).add_(1 - beta1, fp32_grad)
                fp32_exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, fp32_grad, fp32_grad)

                if p.ndim != 1:
                    print(f"[FP8Adam] after mul and add: fp32_exp_avg: {fp32_exp_avg[:2, :2]} \n")
                    print(f"[FP8Adam] after mul and add: fp32_exp_avg_sq: {fp32_exp_avg_sq[:2, :2]} \n")

                if p.ndim != 1:
                    print(f"[FP8Adam] fp32_exp_avg_sq.sqrt(): {fp32_exp_avg_sq.sqrt()[:2, :2]} \n")
                    print(f"[FP8Adam] math.sqrt(bias_correction2)): {math.sqrt(bias_correction2)} \n")
                    print(f"[FP8Adam] group['eps']: {group['eps']} \n")

                denom = (fp32_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])

                if p.ndim != 1:
                    print(f"[FP8Adam] denom: {denom[:2, :2]} \n")

                step_size = group["lr"] / bias_correction1

                if p.ndim != 1:
                    print(f"[FP8Adam] group['lr']: {group['lr']} \n")
                    print(f"[FP8Adam] step_size: {step_size} \n")

                fp32_p = fp32_p - step_size * (fp32_exp_avg / denom)

                # if p.ndim != 1:
                #     print(
                #         f"[FP8Adam] updated_exp_avg_fp8: updated_exp_avg_fp8.data={updated_exp_avg_fp8.data[:2, :2]}, updated_exp_avg_fp8.fp8_meta={updated_exp_avg_fp8.fp8_meta} \n"
                #     )

                # if p.ndim != 1:
                #     print(f"[FP8Adam] updated p_fp32: {fp32_p[:2, :2]} \n")

                # # NOTE: store back fp8
                exp_avg.set_data(fp32_exp_avg)
                exp_avg_sq.set_data(fp32_exp_avg_sq)

                # NOTE: i tried to isinstance(p, FP8Parameter) but it's not working
                # it returns False even though p is FP8Parameter
                self.mappping_fp8_to_master_weight[p] = FP16Tensor(fp32_p, dtype=DTypes.KFLOAT16)

                if p.ndim != 1:
                    self.fp32_p = fp32_p.clone()
                    p.data.set_data(fp32_p)
                    self.updated_p_fp8 = p.data.data

                    # if p.ndim != 1:
                    #     print(
                    #         f"[FP8Adam] updated_p_fp8: updated_p_fp8.data={updated_p_fp8.data[:2, :2]}, updated_p_fp8.fp8_meta={updated_p_fp8.fp8_meta} \n"
                    #     )
                else:
                    # fp16_p = FP16Tensor(fp32_p, dtype=DTypes.KFLOAT16)
                    # p.fp8_meta = fp16_p.fp8_meta
                    fp16_p = fp32_p.to(torch.float16)
                    p.data = fp16_p

                    # p.data = fp16_p
                    if p.ndim != 1:
                        print(f"[FP8Adam] fp32_p: fp32_p.data={fp32_p.data[:2]} \n")
                        print(f"[FP8Adam] fp16_p: fp16_p.data={fp16_p.data[:2]} \n")
                        print(f"[FP8Adam] p: p.data={p.data[:2]} \n")

                # print(f"[FP8Adam] updated_p_fp8: updated_p_fp8.data={updated_p_fp8.data[:2, :2]}, p_fp32_meta={p_fp32_meta} \n")

        self.loggings.append(loggings)
        
    def zero_grad(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                p.grad.zero_()
