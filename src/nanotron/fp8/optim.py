from typing import Any, Dict, List, Tuple, Union

import torch
from torch import nn
from torch.optim import Optimizer

from nanotron import logging
from nanotron._utils.memory import delete_tensor_from_memory
from nanotron.fp8.constants import FP8_DTYPES, FP8LM_RECIPE
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.parameter import FP8Parameter
from nanotron.fp8.recipe import FP8OptimRecipe
from nanotron.fp8.tensor import (
    FP8Tensor,
    FP16Tensor,
    convert_tensor_from_fp8,
    convert_tensor_from_fp16,
)
from nanotron.fp8.utils import compute_stas, is_overflow_underflow_nan
from nanotron.logging import log_rank
from nanotron.parallel.parameters import (
    NanotronParameter,
    get_data_from_param,
    get_data_from_sliced_or_param,
    get_grad_from_parameter,
    set_data_for_sliced_or_param,
    set_grad_none_for_sliced_or_param,
)

logger = logging.get_logger(__name__)


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

    def _get_optim_logs(self):
        from nanotron.scaling.monitor import convert_logs_to_flat_logs

        optim_loggings = {}
        for p in self.loggings:
            param_name = self.params_id_to_param_names[id(p)]
            optim_loggings[param_name] = self.loggings[p]
        return convert_logs_to_flat_logs(optim_loggings)

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
                # data = p.data
                data = get_data_from_param(p)
                assert isinstance(data, torch.Tensor)

                if len(state) == 0:
                    state["step"] = torch.tensor(0.0, dtype=data.dtype)
                    state["exp_avg"] = torch.zeros_like(data, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(data, memory_format=torch.preserve_format)

                loggings[p] = {}

                # assert (p.grad is not None and p.data.grad is not None) is False
                # grad = p.grad if p.grad is not None else p.data.grad
                # grad = get_grad_from_parameter(p)
                grad = get_grad_from_parameter(p)

                from nanotron import constants
                from nanotron.constants import get_debug_save_path

                if constants.is_ready_to_log is True and constants.CONFIG.logging.monitor_model_states is True:
                    # debug_save_path = constants.DEBUG_SAVE_PATH.format(constants.CONFIG.general.run, constants.ITERATION_STEP)
                    debug_save_path = get_debug_save_path(constants.CONFIG.general.run, constants.ITERATION_STEP)

                    torch.save(grad, f"{debug_save_path}/{self.params_id_to_param_names[id(p)]}_before_update_grad.pt")
                    torch.save(
                        data, f"{debug_save_path}/{self.params_id_to_param_names[id(p)]}_before_update_weight.pt"
                    )

                assert isinstance(grad, torch.Tensor)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                if group["weight_decay"] != 0:
                    # grad = grad.add(group["weight_decay"], data)
                    grad = grad + group["weight_decay"] * data

                # Decay the first and second moment running average coefficient
                # exp_avg.mul_(beta1).add_(1 - beta1, grad)
                # exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad.pow(2)

                step = state["step"]
                step += 1
                bias_correction1 = 1 - (beta1**step)
                bias_correction2 = 1 - (beta2**step)

                # exp_avg = exp_avg / bias_correction1
                # exp_avg_sq = exp_avg_sq / bias_correction2

                unbiased_exp_avg = exp_avg / bias_correction1
                unbiased_exp_avg_sq = exp_avg_sq / bias_correction2

                # denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])
                denom = unbiased_exp_avg_sq.sqrt() + group["eps"]
                normalized_grad = unbiased_exp_avg / denom

                lr = group["lr"]
                # p.data.addcdiv_(-step_size, exp_avg, denom)
                # new_data = data - lr * (normalized_grad + (group["weight_decay"] * data))

                if group["weight_decay"] != 0:
                    new_data = data - lr * (normalized_grad + (group["weight_decay"] * data))
                else:
                    new_data = data - lr * normalized_grad

                new_data.requires_grad = True

                # p.data = new_data
                # assert p.data is new_data

                set_data_for_sliced_or_param(p, new_data)
                assert get_data_from_sliced_or_param(p) is new_data

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

        self.loggings = self._get_optim_logs()

        return loss

    def zero_grad(self):
        for group in self.param_groups:
            for p in group["params"]:
                set_grad_none_for_sliced_or_param(p)

                assert p.grad is None
                assert p.data.grad is None


def write_to_file(content, filename="output.txt"):
    """
    Writes content to a file, appending each new piece of content on a new line.

    Args:
    content (str): The text to be written to the file.
    filename (str): The name of the file to write to. Defaults to "output.txt".
    """
    with open(filename, "a") as file:
        file.write(content + "\n")


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
        # accum_dtype: Union[DTypes, torch.dtype] = FP8LM_RECIPE.optim.accum_dtype,
        recipe: FP8OptimRecipe = FP8LM_RECIPE,
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

        # assert accum_dtype in [DTypes] or isinstance(accum_dtype, torch.dtype), f"Please provide an accumulation precision format, accum_dtype: {accum_dtype}"

        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "amsgrad": False}

        super().__init__(params, defaults)

        # TODO(xrsrke): make FP8Adam take a FP8Recipe
        # then retrieve the exp_avg_dtype from the recipe
        self.recipe = recipe
        self.master_weight_dtype = recipe.master_weight_dtype
        self.optim_accum_dtype = recipe.accum_dtype

        # NOTE: torch.Tensor is bias
        self.fp8_weights: List[Union[FP8Parameter, torch.Tensor]] = []
        # NOTE: use to map fp8 param to master weights
        self.mappping_fp8_to_master_weight: Dict[FP8Tensor, Union[FP16Tensor, torch.Tensor]] = {}

        for group in self.param_groups:
            for p in group["params"]:
                data = get_data_from_param(p)
                # NOTE: this parameter we don't convert to FP8, so no need master weight
                if data.__class__ != FP8Tensor:
                    continue

                assert p.dtype == data.dtype

                if isinstance(p, NanotronParameter):
                    raw_data = p.data.orig_data if hasattr(p.data, "orig_data") else p.data
                else:
                    raw_data = p.orig_data if hasattr(p, "orig_data") else p.data

                assert raw_data.dtype in [torch.float32], f"raw_data.dtype={raw_data.dtype}"

                self.mappping_fp8_to_master_weight[p] = self._create_master_weight(raw_data)

                self.fp8_weights.append(p.data)

                delete_tensor_from_memory(raw_data)

                p.orig_data = None
                if hasattr(p.data, "orig_data"):
                    p.data.orig_data = None

        assert len(self.mappping_fp8_to_master_weight) == len(self.fp8_weights)
        # TODO(xrsrke): auto free fp32 weights from memory

        self.loggings = []
        self._is_overflow = False

    def _create_master_weight(self, data):
        if self.master_weight_dtype == DTypes.KFLOAT16:
            master_p = FP16Tensor(data, dtype=DTypes.KFLOAT16)
        elif isinstance(self.master_weight_dtype, torch.dtype):
            master_p = data.to(self.master_weight_dtype) if data.dtype != self.master_weight_dtype else data
        else:
            raise ValueError(f"accum_dtype={self.master_weight_dtype}")
        return master_p

    # def _master_weight_to_accum_weight(self):
    #     pass

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    def _init_optim_states(
        self,
        state: Dict[str, Any],
        p: nn.Parameter,
    ) -> None:
        # TODO(xrsrke): only cast to FP8Tensor if the dtype is FP8
        # state["exp_avg"] = FP8Tensor(torch.zeros(p.data.shape, memory_format=torch.preserve_format), dtype=self.exp_avg_dtype)

        # Exponential moving average of gradient values
        # TODO(xrsrke): maybe initialize a lower precision then cast to FP8Tensor
        # because zeros fp16 = zeros fp32?
        # exp_avg = torch.zeros(p.data.shape, dtype=torch.float32, device="cuda")
        # exp_avg = FP8Tensor(exp_avg, dtype=self.exp_avg_dtype)
        # exp_avg = torch.zeros(p.data.shape, dtype=torch.float32, device="cuda")
        # exp_avg_sq = torch.zeros(p.data.shape, dtype=torch.float32, device="cuda")
        exp_avg = torch.zeros(p.data.shape, dtype=self.optim_accum_dtype, device="cuda")
        exp_avg_sq = torch.zeros(p.data.shape, dtype=self.optim_accum_dtype, device="cuda")

        # state["step"] = torch.tensor(0.0, dtype=torch.float32, device="cuda")
        state["step"] = torch.tensor(0.0, dtype=self.optim_accum_dtype, device="cuda")
        state["exp_avg"] = exp_avg
        state["exp_avg_sq"] = exp_avg_sq

    def _calculate_mean_sqrt_ignoring_nans(self, numerator, denominator):
        # Calculate the division, ignoring division by zero
        division_result = torch.where(denominator != 0, numerator / denominator, torch.zeros_like(numerator))

        # Calculate the mean, ignoring NaN values
        valid_values = division_result[~torch.isnan(division_result)]

        if valid_values.numel() > 0:
            mean_result = valid_values.mean()
            return torch.sqrt(mean_result)
        else:
            raise ValueError("All values are NaN")
            # return torch.tensor(0.0)  # Return 0 if all values are NaN

    def _get_optim_logs(self):
        from nanotron.scaling.monitor import convert_logs_to_flat_logs

        optim_loggings = {}
        for p in self.loggings:
            param_name = self.params_id_to_param_names[id(p)]
            optim_loggings[param_name] = self.loggings[p]
        return convert_logs_to_flat_logs(optim_loggings)

    @torch.no_grad()
    def step(self):
        # NOTE: sanity check the entire params has at least one grad
        # TODO(xrsrke): remove this after debugging
        num_param_has_grads = 0
        for g in self.param_groups:
            for p in g["params"]:
                grad = get_grad_from_parameter(p)
                assert grad is not None
                if p is not None:
                    num_param_has_grads += 1

        assert num_param_has_grads > 0

        self._is_overflow = False
        loggings = {}

        from typing import cast

        from nanotron import constants
        from nanotron.config.fp8_config import FP8Args

        fp8_config = cast(FP8Args, constants.CONFIG.fp8)
        non_fp8_accum_dtype = fp8_config.resid_dtype

        for i, group in enumerate(self.param_groups):
            for p in group["params"]:
                p_name = self.params_id_to_param_names[id(p)]
                loggings[p] = {}
                state = self.state[p]
                if len(state) == 0:
                    self._init_optim_states(state, p)

                data = get_data_from_param(p)
                IS_FP8 = data.__class__ == FP8Tensor

                if hasattr(self, "params_id_to_param_names"):
                    if "0.pp_block.mlp.down_proj.weight" in p_name:
                        assert 1 == 1

                if IS_FP8:
                    assert data.dtype in FP8_DTYPES
                    assert p in self.mappping_fp8_to_master_weight, "FP8Tensor should have a master weight"

                    # NOTE: sanity check
                    if self.master_weight_dtype == DTypes.KFLOAT16:
                        master_data = self.mappping_fp8_to_master_weight[p]
                        fp32_data = convert_tensor_from_fp16(master_data, self.optim_accum_dtype)
                    else:
                        master_data = self.mappping_fp8_to_master_weight[p]
                        fp32_data = (
                            master_data.to(self.self.optim_accum_dtype)
                            if master_data.dtype != self.optim_accum_dtype
                            else master_data
                        )

                    grad = get_grad_from_parameter(p)
                    assert grad is not None
                    assert grad.dtype in FP8_DTYPES
                    fp32_grad = convert_tensor_from_fp8(grad, grad.fp8_meta, self.optim_accum_dtype)
                else:
                    assert (
                        data.dtype == non_fp8_accum_dtype
                    ), f"data.dtype={data.dtype}, non_fp8_accum_dtype={non_fp8_accum_dtype}"
                    fp32_data = data.to(self.optim_accum_dtype) if data.dtype != self.optim_accum_dtype else data
                    grad = get_grad_from_parameter(p)
                    assert grad is not None
                    assert grad.dtype == non_fp8_accum_dtype
                    fp32_grad = grad.to(self.optim_accum_dtype) if grad.dtype != self.optim_accum_dtype else grad

                assert fp32_data.dtype == self.optim_accum_dtype
                assert fp32_grad.dtype == self.optim_accum_dtype

                if is_overflow_underflow_nan(fp32_grad):
                    self._is_overflow = True
                    raise ValueError("Overflow, underflow, or NaN detected in the gradients")

                fp32_exp_avg, fp32_exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                assert fp32_exp_avg.dtype == self.optim_accum_dtype
                assert fp32_exp_avg_sq.dtype == self.optim_accum_dtype

                beta1, beta2 = group["betas"]
                lr = group["lr"]
                step = state["step"]
                step += 1

                if hasattr(self, "params_id_to_param_names"):
                    # param_name = self.params_id_to_param_names[id(p)]
                    if "0.pp_block.mlp.down_proj.weight" in p_name or "lm_head" in p_name:
                        from nanotron import constants

                        write_to_file(
                            f"step={constants.ITERATION_STEP}, param={p_name}, lr={group['lr']}, initial_lr={group['initial_lr']}",
                            filename="/fsx/phuc/temp/temp3_env_for_fp8/nanotron/lr_logs.txt",
                        )

                fp32_exp_avg = beta1 * fp32_exp_avg + (1 - beta1) * fp32_grad
                fp32_exp_avg_sq = beta2 * fp32_exp_avg_sq + (1 - beta2) * fp32_grad.pow(2)

                bias_correction1 = 1 / (1 - (beta1**step))
                bias_correction2 = 1 / (1 - (beta2**step))

                unbiased_fp32_exp_avg = fp32_exp_avg * bias_correction1
                unbiased_fp32_exp_avg_sq = fp32_exp_avg_sq * bias_correction2

                denom = unbiased_fp32_exp_avg_sq.sqrt() + group["eps"]
                normalized_grad = unbiased_fp32_exp_avg / denom

                # if step >= 10:
                #     if "0.pp_block.mlp.down_proj.weight" in self.params_id_to_param_names[id(p)]:
                #         assert 1 == 1

                #     # rms = (fp32_grad.pow(2) / unbiased_fp32_exp_avg_sq).mean().sqrt()
                #     rms = self._calculate_mean_sqrt_ignoring_nans(fp32_grad, unbiased_fp32_exp_avg_sq).sqrt()
                # else:
                #     rms = torch.tensor(1.0, dtype=self.optim_accum_dtype, device="cuda")

                if "0.pp_block.mlp.down_proj.weight" in p_name:
                    assert 1 == 1

                if step >= 10:
                    if "0.pp_block.mlp.down_proj.weight" in p_name:
                        assert 1 == 1

                rms = self._calculate_mean_sqrt_ignoring_nans(fp32_grad.pow(2), unbiased_fp32_exp_avg_sq)

                # log_rank(
                #     f"[Gradient clipping] The RMS is {rms}",  # noqa
                #     logger=logger,
                #     level=logging.INFO,
                #     rank=0,
                # )

                if constants.CONFIG.optimizer.update_clipping is True:
                    update_lr = lr / torch.max(torch.tensor(1.0, dtype=self.optim_accum_dtype, device="cuda"), rms)
                    if rms > 1:
                        log_rank(
                            f"[Gradient clipping] param_name={p_name}, grad_norm: {fp32_grad.norm(p=2)}, RMS is {rms}, original lr is {lr}, new lr is {update_lr}",  # noqa
                            logger=logger,
                            level=logging.INFO,
                            rank=0,
                        )
                else:
                    update_lr = lr

                weight_decay_factor = group["weight_decay"] if data.ndim >= 2 else 0.

                if weight_decay_factor != 0:
                    new_fp32_data = fp32_data - update_lr * (normalized_grad + weight_decay_factor * fp32_data)
                else:
                    new_fp32_data = fp32_data - update_lr * normalized_grad

                if IS_FP8:
                    # self.mappping_fp8_to_master_weight[p] = FP16Tensor(new_fp32_data, dtype=DTypes.KFLOAT16)
                    self.mappping_fp8_to_master_weight[p] = self._create_master_weight(new_fp32_data)
                    p.data.set_data(new_fp32_data)

                    # NOTE: SANITY CHECK
                    if self.master_weight_dtype == DTypes.KFLOAT16:
                        _dequant_master_data = convert_tensor_from_fp16(
                            self.mappping_fp8_to_master_weight[p], DTypes.KFLOAT16, torch.float32
                        )
                        torch.testing.assert_allclose(_dequant_master_data, new_fp32_data)

                    _quant_new_fp32_data = get_data_from_param(p)
                    _dequant_new_fp32_data = convert_tensor_from_fp8(
                        _quant_new_fp32_data, _quant_new_fp32_data.fp8_meta, torch.float32
                    )
                    from nanotron.fp8.constants import FP8_WEIGHT_ATOL_THRESHOLD, FP8_WEIGHT_RTOL_THRESHOLD

                    torch.testing.assert_allclose(
                        _dequant_new_fp32_data,
                        new_fp32_data,
                        rtol=FP8_WEIGHT_RTOL_THRESHOLD,
                        atol=FP8_WEIGHT_ATOL_THRESHOLD,
                    )

                else:
                    new_fp16 = (
                        new_fp32_data.to(non_fp8_accum_dtype)
                        if new_fp32_data.dtype != non_fp8_accum_dtype
                        else new_fp32_data
                    )
                    new_fp16.requires_grad = True
                    p.data = new_fp16
                    assert get_data_from_param(p) is new_fp16

                    torch.testing.assert_allclose(get_data_from_param(p), new_fp16)

                # delete_tensor_from_memory(new_fp32_data)
                # delete_tensor_from_memory(denom)

                state["step"] = step
                state["exp_avg"] = fp32_exp_avg
                state["exp_avg_sq"] = fp32_exp_avg_sq

                assert state["step"] == step
                assert state["exp_avg"] is fp32_exp_avg
                assert state["exp_avg_sq"] is fp32_exp_avg_sq

                loggings[p]["step"] = {"value": step}
                loggings[p]["group:lr"] = {"value": lr}
                loggings[p]["group:eps"] = {"value": group["eps"]}
                loggings[p]["group:beta1"] = {"value": beta1}
                loggings[p]["group:beta2"] = {"value": beta2}

                loggings[p]["bias_correction1"] = {"value": bias_correction1}
                loggings[p]["bias_correction2"] = {"value": bias_correction2}
                loggings[p]["fp32_exp_avg"] = compute_stas(fp32_exp_avg)
                loggings[p]["fp32_exp_avg_sq"] = compute_stas(fp32_exp_avg_sq)

                loggings[p]["normalized_grad"] = compute_stas(normalized_grad)
                loggings[p]["denom"] = compute_stas(denom)
                loggings[p]["grad_rms"] = {"value": rms}
                loggings[p]["update_lr"] = {"value": update_lr}

                loggings[p]["fp32_p"] = compute_stas(fp32_data)
                loggings[p]["fp32_grad"] = compute_stas(fp32_grad)

        self.loggings = loggings

        self.loggings = self._get_optim_logs()

        # NOTE: grouping statistics by layer
        assert 1 == 1

    def zero_grad(self):
        for group in self.param_groups:
            for p in group["params"]:
                # NOTE: take the assumption that nanotron requires all parameters to have gradients
                set_grad_none_for_sliced_or_param(p)

                assert p.grad is None
                assert p.data.grad is None
