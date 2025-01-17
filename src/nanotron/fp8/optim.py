from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from torch.optim import Optimizer

from nanotron import logging
from nanotron.fp8.constants import FP8LM_RECIPE
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.recipe import FP8OptimRecipe
from nanotron.fp8.tensor import (
    FP8Tensor,
    FP16Tensor,
    convert_tensor_from_fp8,
    convert_tensor_from_fp16,
)
from nanotron.fp8.utils import is_overflow_underflow_nan
from nanotron.logging import log_rank

logger = logging.get_logger(__name__)


class FP8AdamW(Optimizer):
    """
    FP8 AdamW optimizer.
    """

    def __init__(
        self,
        params: List[nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        recipe: FP8OptimRecipe = FP8LM_RECIPE,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "amsgrad": False}

        super().__init__(params, defaults)

        # TODO(xrsrke): make FP8Adam take a FP8Recipe
        # then retrieve the exp_avg_dtype from the recipe
        self.recipe = recipe
        self.master_weight_dtype = recipe.master_weight_dtype
        self.optim_accum_dtype = recipe.accum_dtype

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

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    # TODO(xrsrke): this is similar with master weight func, remove this
    def _quantize_optim_state(self, tensor, dtype):
        if dtype == DTypes.FP8E4M3 or dtype == DTypes.FP8E5M2:
            tensor = FP8Tensor(tensor, dtype=dtype)
        elif dtype == DTypes.KFLOAT16:
            tensor = FP16Tensor(tensor, dtype=DTypes.KFLOAT16)
        elif isinstance(dtype, torch.dtype):
            tensor = tensor.to(dtype)
        else:
            raise ValueError(f"supported dtype={dtype}")
        return tensor

    def _init_optim_states(
        self,
        state: Dict[str, Any],
        p: nn.Parameter,
    ) -> None:
        # TODO(xrsrke): could we initialize these at a lower precision
        # than the accumulation precision (eg: float32) because
        # these are just zero tensors anyway?
        exp_avg = torch.zeros_like(p.data, dtype=self.optim_accum_dtype)
        exp_avg_sq = torch.zeros_like(p.data, dtype=self.optim_accum_dtype)

        exp_avg = self._quantize_optim_state(exp_avg, self.recipe.exp_avg_dtype)
        exp_avg_sq = self._quantize_optim_state(exp_avg_sq, self.recipe.exp_avg_sq_dtype)

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

    # def _get_optim_logs(self):
    #     from nanotron.scaling.monitor import convert_logs_to_flat_logs

    #     optim_loggings = {}
    #     for p in self.loggings:
    #         param_name = self.params_id_to_param_names[id(p)]
    #         optim_loggings[param_name] = self.loggings[p]
    #     return convert_logs_to_flat_logs(optim_loggings)

    def _dequantize_optim_state(self, state):
        if state.__class__ == FP8Tensor:
            fp32_state = convert_tensor_from_fp8(state, state.fp8_meta, self.optim_accum_dtype)
        elif state.__class__ == FP16Tensor:
            fp32_state = convert_tensor_from_fp16(state, self.optim_accum_dtype)
        elif state.dtype == self.optim_accum_dtype:
            fp32_state = state
        elif isinstance(state.dtype, torch.dtype):
            fp32_state = state.to(self.optim_accum_dtype) if state.dtype != self.optim_accum_dtype else state

        return fp32_state

    @torch.no_grad()
    def step(self, closure=None):
        # NOTE: sanity check the entire params has at least one grad
        # TODO(xrsrke): remove this after debugging
        from typing import cast

        from nanotron import constants
        from nanotron.config.fp8_config import FP8Args

        cast(FP8Args, constants.CONFIG.fp8)

        for i, group in enumerate(self.param_groups):
            for p in group["params"]:

                if not isinstance(p.data, FP8Tensor) and p.requires_grad is False:
                    continue

                assert p.grad is not None

                state = self.state[p]
                if len(state) == 0:
                    self._init_optim_states(state, p)

                # NOTE: Case 1: With gradient accumulator => the grad is already in the correct dtype
                # Case 2: Without gradient accumulator =>
                # 2.1 Non-FP8 parameter => cast the grad to the correct dtype
                # 2.2 FP8 parameter => dequantize the grad to the correct dtype

                fp32_grad = p.grad
                fp32_data = p.data
                assert fp32_grad.dtype == self.optim_accum_dtype
                assert p.data.dtype == torch.float32

                if is_overflow_underflow_nan(fp32_grad):
                    self._is_overflow = True

                    if constants.CONFIG.fp8.skip_param_update_if_nan is True:
                        log_rank(
                            f"[Optim] param_name, skipping update due to overflow/underflow/nan",  # noqa
                            logger=logger,
                            level=logging.INFO,
                        )
                        continue
                    else:
                        raise ValueError("Overflow, underflow, or NaN detected in the gradients")

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                fp32_exp_avg = self._dequantize_optim_state(exp_avg)
                fp32_exp_avg_sq = self._dequantize_optim_state(exp_avg_sq)

                assert fp32_exp_avg.dtype == self.optim_accum_dtype
                assert fp32_exp_avg_sq.dtype == self.optim_accum_dtype

                beta1, beta2 = group["betas"]
                lr = group["lr"]
                step = state["step"]
                step += 1

                fp32_exp_avg = beta1 * fp32_exp_avg + (1 - beta1) * fp32_grad
                fp32_exp_avg_sq = beta2 * fp32_exp_avg_sq + (1 - beta2) * fp32_grad.pow(2)

                bias_correction1 = 1 / (1 - (beta1**step))
                bias_correction2 = 1 / (1 - (beta2**step))

                unbiased_fp32_exp_avg = fp32_exp_avg * bias_correction1
                unbiased_fp32_exp_avg_sq = fp32_exp_avg_sq * bias_correction2

                denom = unbiased_fp32_exp_avg_sq.sqrt() + group["eps"]
                normalized_grad = unbiased_fp32_exp_avg / denom

                if constants.CONFIG.fp8.update_clipping is True:
                    rms = self._calculate_mean_sqrt_ignoring_nans(
                        fp32_grad.pow(2),
                        torch.max(
                            unbiased_fp32_exp_avg_sq,
                            torch.tensor(group["eps"], dtype=self.optim_accum_dtype, device="cuda").pow(2),
                        ),
                    )

                    if rms > 1:
                        # NOTE: only scale down the lr, not scale it up
                        update_lr = lr / torch.max(torch.tensor(1.0, dtype=self.optim_accum_dtype, device="cuda"), rms)
                        log_rank(
                            f"[Gradient clipping] param_name=, grad_norm: {fp32_grad.norm(p=2)}, RMS is {rms}, original lr is {lr}, new lr is {update_lr}",  # noqa
                            logger=logger,
                            level=logging.INFO,
                            rank=0,
                        )
                    else:
                        update_lr = lr
                else:
                    update_lr = lr

                # NOTE: keep weight decay for biases
                # TODO(xrsrke): we should explicitly set weight_decay_factor to 0 for biases
                # in optimizer's param_groups
                weight_decay_factor = group["weight_decay"] if p.data.ndim >= 2 else 0.0

                if weight_decay_factor != 0:
                    fp32_new_changes_from_grad = update_lr * normalized_grad
                    fp32_weight_decay_grad = weight_decay_factor * fp32_data

                    if constants.CONFIG.fp8.weight_decay_without_lr_decay is False:
                        fp32_new_changes_from_weight_decay = update_lr * fp32_weight_decay_grad
                    else:
                        fp32_new_changes_from_weight_decay = (
                            constants.CONFIG.optimizer.learning_rate_scheduler.learning_rate * fp32_weight_decay_grad
                        )
                else:
                    fp32_new_changes_from_grad = update_lr * normalized_grad
                    fp32_new_changes_from_weight_decay = 0

                fp32_new_changes_in_p = fp32_new_changes_from_grad + fp32_new_changes_from_weight_decay
                new_fp32_data = fp32_data - fp32_new_changes_in_p

                p.data = new_fp32_data

                exp_avg = self._quantize_optim_state(fp32_exp_avg, self.recipe.exp_avg_dtype)
                exp_avg_sq = self._quantize_optim_state(fp32_exp_avg_sq, self.recipe.exp_avg_sq_dtype)

                state["step"] = step
                state["exp_avg"] = exp_avg
                state["exp_avg_sq"] = exp_avg_sq

                assert state["step"] == step
                assert state["exp_avg"] is exp_avg
                assert state["exp_avg_sq"] is exp_avg_sq

                # NOTE: remove this shit
                # if constants.is_ready_to_log is True:
                #     loggings[p]["step"] = {"value": step}
                #     loggings[p]["group:lr"] = {"value": lr}
                #     loggings[p]["group:eps"] = {"value": group["eps"]}
                #     loggings[p]["group:beta1"] = {"value": beta1}
                #     loggings[p]["group:beta2"] = {"value": beta2}

                #     loggings[p]["bias_correction1"] = {"value": bias_correction1}
                #     loggings[p]["bias_correction2"] = {"value": bias_correction2}
                #     loggings[p]["fp32_exp_avg"] = compute_stas(fp32_exp_avg)
                #     loggings[p]["fp32_exp_avg_sq"] = compute_stas(fp32_exp_avg_sq)

                #     loggings[p]["normalized_grad"] = compute_stas(normalized_grad)

                #     if fp8_config.adam_atan2 is False:
                #         loggings[p]["denom"] = compute_stas(denom)

                #     loggings[p]["update_lr"] = {"value": update_lr}

                #     loggings[p]["fp32_p"] = compute_stas(fp32_data)
                #     loggings[p]["fp32_new_changes_in_p"] = {
                #         # "abs_total": fp32_new_changes_in_p.abs().sum(),
                #         # "abs_mean": fp32_new_changes_in_p.abs().mean(),
                #         "rms": fp32_new_changes_in_p.pow(2)
                #         .mean()
                #         .sqrt(),
                #     }
                #     loggings[p]["fp32_new_changes_from_grad"] = {
                #         "rms": fp32_new_changes_from_grad.pow(2).mean().sqrt(),
                #     }

                #     p_norm = fp32_data.norm()

                #     loggings[p]["fp32_grad"] = compute_stas(fp32_grad)
                #     loggings[p]["update_lr"] = {"value": update_lr}
                #     loggings[p]["weight_norm_and_normalized_grad_norm_ratio"] = {
                #         "value": p_norm / fp32_new_changes_from_grad.norm()
                #     }
                #     loggings[p]["weight_norm_and_weight_update_norm_ratio"] = {
                #         "value": p_norm / fp32_new_changes_in_p.norm()
                #     }

                #     if weight_decay_factor != 0:
                #         loggings[p]["fp32_new_changes_from_weight_decay"] = {
                #             "rms": fp32_new_changes_from_weight_decay.pow(2).mean().sqrt(),
                #         }
                #         loggings[p]["weight_norm_and_weight_decay_grad_norm_ratio"] = {
                #             "value": p_norm / fp32_weight_decay_grad.norm()
                #         }

                #     if constants.CONFIG.fp8.update_clipping is True:
                #         loggings[p]["grad_rms"] = {"value": rms}

        # if constants.is_ready_to_log is True:
        #     self.loggings = loggings
        #     self.loggings = self._get_optim_logs()

    def zero_grad(self):
        for group in self.param_groups:
            for p in group["params"]:
                # NOTE: take the assumption that nanotron requires all parameters to have gradients
                p.grad = None

                assert p.grad is None
                assert p.data.grad is None
