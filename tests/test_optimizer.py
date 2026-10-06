from typing import Union

import pytest
import torch
from helpers.utils import init_distributed, rerun_if_address_is_in_use
from nanotron import distributed as dist
from nanotron.config import ModelArgs, RandomInit, SpectralMupInit
from nanotron.helpers import get_custom_lr_for_named_parameters, get_custom_weight_decay_for_named_parameters
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import NanotronParameter
from nanotron.parallel.tied_parameters import tie_parameters
from nanotron.scaling.parametrization import ParametrizationMethod
from torch import nn

from tests.helpers.llama_helper import TINY_LLAMA_CONFIG, create_llama_from_config, get_llama_training_config


@pytest.mark.parametrize("tp,dp,pp", [(1, 1, 1), (2, 1, 1), (1, 1, 2), (2, 1, 2)])
@pytest.mark.parametrize(
    "parametrization_method", [ParametrizationMethod.STANDARD, ParametrizationMethod.SPECTRAL_MUP]
)
@pytest.mark.skip
@rerun_if_address_is_in_use()
def test_get_custom_lr(tp: int, dp: int, pp: int, parametrization_method: ParametrizationMethod):
    LR = 1e-3

    if parametrization_method == ParametrizationMethod.STANDARD:
        init_method = RandomInit(std=1.0)
    elif parametrization_method == ParametrizationMethod.SPECTRAL_MUP:
        init_method = SpectralMupInit(use_mup=True)

    init_distributed(tp=tp, dp=dp, pp=pp)(_test_get_custom_lr)(
        lr=LR,
        init_method=init_method,
        parametrization_method=parametrization_method,
    )


def _test_get_custom_lr(
    parallel_context: ParallelContext,
    lr: float,
    init_method: Union[RandomInit, SpectralMupInit],
    parametrization_method: ParametrizationMethod,
):
    model_args = ModelArgs(init_method=init_method, model_config=TINY_LLAMA_CONFIG)
    config = get_llama_training_config(model_args)
    llama = create_llama_from_config(
        model_config=TINY_LLAMA_CONFIG,
        device=torch.device("cuda"),
        parallel_context=parallel_context,
    )
    llama.init_model_randomly(config=config, init_method=parametrization_method)
    named_parameters = list(llama.get_named_params_with_correct_tied())

    if len(named_parameters) == 0:
        # NOTE: some pp ranks don't have any parameters
        return

    named_param_groups = get_custom_lr_for_named_parameters(
        parametrization_method=parametrization_method, lr=lr, named_parameters=named_parameters, model=llama
    )

    assert len(named_param_groups) == len(named_parameters)
    assert all(isinstance(named_param_group["lr"], float) for named_param_group in named_param_groups)
    assert all(isinstance(named_param_group["named_params"], list) for named_param_group in named_param_groups)

    is_all_lr_the_same = parametrization_method == ParametrizationMethod.STANDARD
    assert all(named_param_group["lr"] == lr for named_param_group in named_param_groups) is is_all_lr_the_same


@pytest.mark.parametrize("tie_embeddings", [True, False])
@rerun_if_address_is_in_use()
def test_get_custom_weight_decay_exclude_named_params(tie_embeddings: bool):
    init_distributed(tp=1, dp=1, pp=1)(_test_get_custom_weight_decay_exclude_named_params)(
        tie_embeddings=tie_embeddings
    )


def _test_get_custom_weight_decay_exclude_named_params(parallel_context: ParallelContext, tie_embeddings: bool):
    weight_decay = 0.1
    model = nn.ModuleDict(
        {
            "embed": nn.Linear(10, 10, bias=False),
            "dense": nn.Linear(10, 10, bias=False),
            "lm_head": nn.Linear(10, 10, bias=False),
        }
    )
    for module in model.values():
        module.weight = NanotronParameter(module.weight)
    if tie_embeddings:
        tie_parameters(
            root_module=model,
            ties=[("embed.weight", (0,)), ("lm_head.weight", (0,))],
            parallel_context=parallel_context,
            reduce_op=dist.ReduceOp.SUM,
        )

    module_id_to_prefix = {id(module): f"{module_name}." for module_name, module in model.named_modules()}
    module_id_to_prefix[id(model)] = ""
    named_parameters = [
        (
            param.get_tied_info().get_full_name_from_module_id_to_prefix(module_id_to_prefix=module_id_to_prefix)
            if param.is_tied
            else name,
            param,
        )
        for name, param in model.named_parameters()
    ]

    named_param_groups = get_custom_weight_decay_for_named_parameters(
        named_parameters=named_parameters,
        model=model,
        module_id_to_prefix=module_id_to_prefix,
        weight_decay=weight_decay,
        exclude_named_params=["embed.*"],
    )

    name_to_weight_decay = {group["named_params"][0][0]: group["weight_decay"] for group in named_param_groups}
    expected = {"embed.weight": 0.0, "dense.weight": weight_decay}
    if not tie_embeddings:
        expected["lm_head.weight"] = weight_decay
    assert name_to_weight_decay == expected

    parallel_context.destroy()
