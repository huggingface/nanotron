from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from torch.optim import Optimizer

from nanotron import logging

# from nanotron._utils.memory import delete_tensor_from_memory
from nanotron.fp8.constants import FP8_DTYPES, FP8LM_RECIPE
from nanotron.fp8.dtypes import DTypes
from nanotron.fp8.recipe import FP8OptimRecipe
from nanotron.fp8.tensor import (
    FP8Tensor,
    FP16Tensor,
    convert_tensor_from_fp8,
    convert_tensor_from_fp16,
)
from nanotron.fp8.utils import compute_stas, is_overflow_underflow_nan
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

        # NOTE: torch.Tensor is bias
        # self.fp8_weights: List[Union[FP8Parameter, torch.Tensor]] = []

        # NOTE: move to gradient accumulator
        # # NOTE: create master weights for FP8 Parameter
        # self.mappping_fp8_to_master_weight: Dict[str, Union[FP16Tensor, torch.Tensor]] = {}

        # for group in self.param_groups:
        #     for p in group["params"]:
        #         # data = get_data_from_param(p)
        #         # if p.data.__class__ != FP8Tensor:
        #         #     continue
        #         # # NOTE: this parameter we don't convert to FP8, so no need master weight
        #         # if not isinstance(p.data, FP8Tensor):
        #         #     continue

        #         assert 1 == 1
        #         # if p._is_future_fp8 is not True:
        #         #     continue
        #         if not isinstance(p.data, FP8Tensor):
        #             continue

        #         # assert p.dtype == data.dtype

        #         # if isinstance(p, NanotronParameter):
        #         #     raw_data = p.data.orig_data if hasattr(p.data, "orig_data") else p.data
        #         # else:
        #         #     raw_data = p.orig_data if hasattr(p, "orig_data") else p.data
        #         # assert raw_data.dtype in [torch.float32], f"raw_data.dtype={raw_data.dtype}"

        #         assert p.data.dtype in [torch.float32], f"raw_data.dtype={p.data.dtype}"
        #         self.mappping_fp8_to_master_weight[hash(p)] = self._create_master_weight(p.data)

        #         # self.fp8_weights.append(p.data)

        #         # delete_tensor_from_memory(raw_data)

        #         # p.orig_data = None
        #         # if hasattr(p.data, "orig_data"):
        #         #     p.data.orig_data = None

        # # assert len(self.mappping_fp8_to_master_weight) == len(self.fp8_weights)
        # # TODO(xrsrke): auto free fp32 weights from memory

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

        # if constants.CONFIG.fp8.run_fp8_sanity_check is True:
        #     num_param_has_grads = 0
        #     for g in self.param_groups:
        #         for p in g["params"]:
        #             grad = get_grad_from_parameter(p)
        #             assert grad is not None
        #             if p is not None:
        #                 num_param_has_grads += 1
        #     assert num_param_has_grads > 0

        self._is_overflow = False
        loggings = {}

        fp8_config = cast(FP8Args, constants.CONFIG.fp8)
        non_fp8_accum_dtype = fp8_config.resid_dtype

        # from nanotron.helpers import get_accum_grad, set_accum_grad

        for i, group in enumerate(self.param_groups):
            for p in group["params"]:

                if not isinstance(p.data, FP8Tensor) and p.requires_grad is False:
                    continue

                assert p.grad is not None

                # p_name = self.params_id_to_param_names[id(p)]
                # loggings[p] = {}
                state = self.state[p]
                if len(state) == 0:
                    self._init_optim_states(state, p)

                # data = get_data_from_param(p)
                # IS_FP8 = data.__class__ == FP8Tensor

                # NOTE: if use gradient accumulation, after the backward pass
                # we set the param.grad to None, so we need to retrieve it from accumulator

                # if constants.CONFIG.optimizer.accumulate_grad_in_fp32 is True:
                #     # fp32_grad = self.grad_accumulator.get_grad_buffer(name=p_name)

                #     # if "model.decoder.8.pp_block.attn_layer_scale" in p_name:
                #     #     assert 1 == 1

                #     # if constants.CONFIG.fp8.is_save_grad_for_accum_debugging is True:
                #     #     from nanotron.helpers import create_folder_and_save_tensor

                #     #     create_folder_and_save_tensor(
                #     #         fp32_grad,
                #     #         f"/fsx/phuc/temp/temp3_env_for_fp8/nanotron/debug_accum/{constants.CONFIG.general.run}/aggr_grads/{p_name}.pt",
                #     #     )
                #     raise NotImplementedError("accumulate_grad_in_fp32 is not implemented")
                # else:
                #     if isinstance(p.data, FP8Tensor):
                #         if constants.CONFIG.fp8.is_directly_keep_accum_grad_of_fp8 is True:
                #             # fp32_grad = constants.ACCUM_GRADS[p_name]
                #             # grad = get_accum_grad(p_name)
                #             # fp32_grad = (
                #             #     grad.to(self.optim_accum_dtype) if grad.dtype != self.optim_accum_dtype else grad
                #             # )
                #             # assert fp32_grad.dtype == torch.float32

                #             # # constants.ACCUM_GRADS[p_name] = None
                #             # set_accum_grad(p_name, None)
                #             raise NotImplementedError("is_directly_keep_accum_grad_of_fp8 is not implemented")
                #         else:
                #             assert p.grad.dtype in FP8_DTYPES
                #             fp32_grad = convert_tensor_from_fp8(p.grad, p.grad.fp8_meta, self.optim_accum_dtype)
                #     else:
                #         # grad = get_grad_from_parameter(p)

                #         # assert grad is not None
                #         assert p.grad.dtype == non_fp8_accum_dtype

                #         fp32_grad = p.grad.to(self.optim_accum_dtype)

                # NOTE: Case 1: With gradient accumulator => the grad is already in the correct dtype
                # Case 2: Without gradient accumulator =>
                # 2.1 Non-FP8 parameter => cast the grad to the correct dtype
                # 2.2 FP8 parameter => dequantize the grad to the correct dtype
                grad = p.grad
                if isinstance(p.data, FP8Tensor):
                    fp32_grad = convert_tensor_from_fp8(grad, grad.fp8_meta, self.optim_accum_dtype)
                else:
                    fp32_grad = grad.to(self.optim_accum_dtype)

                assert fp32_grad.dtype == self.optim_accum_dtype

                if isinstance(p.data, FP8Tensor):
                    assert p.data.dtype in FP8_DTYPES
                    assert hash(p) in self.mappping_fp8_to_master_weight, "Can't find master weight for FP8 parameter"

                    master_data = self.mappping_fp8_to_master_weight[hash(p)]
                    if self.master_weight_dtype == DTypes.KFLOAT16:
                        fp32_data = convert_tensor_from_fp16(master_data, self.optim_accum_dtype)
                    else:
                        fp32_data = master_data.to(self.optim_accum_dtype)
                else:
                    assert (
                        p.data.dtype == non_fp8_accum_dtype
                    ), f"data.dtype={p.data.dtype}, non_fp8_accum_dtype={non_fp8_accum_dtype}"
                    fp32_data = p.data.to(self.optim_accum_dtype)

                assert fp32_data.dtype == self.optim_accum_dtype

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

                if fp8_config.adam_atan2 is True:
                    from torch import atan2

                    adam_atan2_lambda = fp8_config.adam_atan2_lambda
                    normalized_grad = adam_atan2_lambda * atan2(
                        unbiased_fp32_exp_avg, adam_atan2_lambda * unbiased_fp32_exp_avg_sq.sqrt()
                    )

                    rms = self._calculate_mean_sqrt_ignoring_nans(
                        fp32_grad.pow(2),
                        unbiased_fp32_exp_avg_sq,
                    )
                else:
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

                    # fp32_new_changes_in_p = update_lr * normalized_grad

                fp32_new_changes_in_p = fp32_new_changes_from_grad + fp32_new_changes_from_weight_decay
                new_fp32_data = fp32_data - fp32_new_changes_in_p

                if isinstance(p.data, FP8Tensor):
                    sync_amax_in_weight = fp8_config.sync_amax_in_weight

                    self.mappping_fp8_to_master_weight[p] = self._create_master_weight(new_fp32_data)
                    p.data.set_data(new_fp32_data, sync=sync_amax_in_weight)

                    # NOTE: SANITY CHECK
                    if constants.CONFIG.fp8.run_fp8_sanity_check is True:
                        if self.master_weight_dtype == DTypes.KFLOAT16:
                            _dequant_master_data = convert_tensor_from_fp16(
                                self.mappping_fp8_to_master_weight[p], DTypes.KFLOAT16, torch.float32
                            )
                            torch.testing.assert_allclose(_dequant_master_data, new_fp32_data)

                        # _quant_new_fp32_data = get_data_from_param(p)
                        _quant_new_fp32_data = p.data
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
                    if constants.CONFIG.fp8.stochastic_rounding is True:
                        raise NotImplementedError("stochastic_rounding is not implemented")
                        assert non_fp8_accum_dtype is torch.bfloat16, "only support stochastic rounding for bfloat16"
                        new_fp16 = torch.full_like(new_fp32_data, 0.0, dtype=non_fp8_accum_dtype)
                        copy_stochastic_(target=new_fp16, source=new_fp32_data)
                    else:
                        # new_fp16 = (
                        #     new_fp32_data.to(non_fp8_accum_dtype)
                        #     if new_fp32_data.dtype != non_fp8_accum_dtype
                        #     else new_fp32_data
                        # )
                        new_fp16 = new_fp32_data.to(non_fp8_accum_dtype)

                    # new_fp16.requires_grad = True
                    # p.data = new_fp16

                    # assert p.data is new_fp16

                    # if constants.CONFIG.fp8.run_fp8_sanity_check is True:
                    #     # torch.testing.assert_allclose(get_data_from_param(p), new_fp16)
                    #     torch.testing.assert_allclose(p.data, new_fp16)

                exp_avg = self._quantize_optim_state(fp32_exp_avg, self.recipe.exp_avg_dtype)
                exp_avg_sq = self._quantize_optim_state(fp32_exp_avg_sq, self.recipe.exp_avg_sq_dtype)

                state["step"] = step
                state["exp_avg"] = exp_avg
                state["exp_avg_sq"] = exp_avg_sq

                assert state["step"] == step
                assert state["exp_avg"] is exp_avg
                assert state["exp_avg_sq"] is exp_avg_sq

                # NOTE: remove this shit
                if constants.is_ready_to_log is True:
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

                    if fp8_config.adam_atan2 is False:
                        loggings[p]["denom"] = compute_stas(denom)

                    loggings[p]["update_lr"] = {"value": update_lr}

                    loggings[p]["fp32_p"] = compute_stas(fp32_data)
                    loggings[p]["fp32_new_changes_in_p"] = {
                        # "abs_total": fp32_new_changes_in_p.abs().sum(),
                        # "abs_mean": fp32_new_changes_in_p.abs().mean(),
                        "rms": fp32_new_changes_in_p.pow(2)
                        .mean()
                        .sqrt(),
                    }
                    loggings[p]["fp32_new_changes_from_grad"] = {
                        "rms": fp32_new_changes_from_grad.pow(2).mean().sqrt(),
                    }

                    p_norm = fp32_data.norm()

                    loggings[p]["fp32_grad"] = compute_stas(fp32_grad)
                    loggings[p]["update_lr"] = {"value": update_lr}
                    loggings[p]["weight_norm_and_normalized_grad_norm_ratio"] = {
                        "value": p_norm / fp32_new_changes_from_grad.norm()
                    }
                    loggings[p]["weight_norm_and_weight_update_norm_ratio"] = {
                        "value": p_norm / fp32_new_changes_in_p.norm()
                    }

                    if weight_decay_factor != 0:
                        loggings[p]["fp32_new_changes_from_weight_decay"] = {
                            "rms": fp32_new_changes_from_weight_decay.pow(2).mean().sqrt(),
                        }
                        loggings[p]["weight_norm_and_weight_decay_grad_norm_ratio"] = {
                            "value": p_norm / fp32_weight_decay_grad.norm()
                        }

                    if constants.CONFIG.fp8.update_clipping is True:
                        loggings[p]["grad_rms"] = {"value": rms}

        # if constants.is_ready_to_log is True:
        #     self.loggings = loggings
        #     self.loggings = self._get_optim_logs()

    def zero_grad(self):
        for group in self.param_groups:
            for p in group["params"]:
                # NOTE: take the assumption that nanotron requires all parameters to have gradients
                # set_grad_none_for_sliced_or_param(p)
                p.grad = None

                assert p.grad is None
                assert p.data.grad is None
