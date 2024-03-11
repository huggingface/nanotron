# ruff: noqa: E402
"""
Converts a HF model from (https://huggingface.co/state-spaces/) to a Brrr model

Command:
    torchrun --nproc_per_node=1 convert_hf_to_nanotron.py --model 130M  --save_path nanotron-weights
"""
import argparse
import torch
import yaml
from pathlib import Path
from tqdm import tqdm
from typing import Dict
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
import lovely_tensors as lt; lt.monkey_patch()

from nanotron.config import (
    AllForwardAllBackwardPipelineEngine,
    ParallelismArgs,
    TensorParallelLinearMode,
)
from config import MambaModelConfig, MambaConfig, MambaInit
from nanotron.config import GeneralArgs, ModelArgs, TokenizerArgs

from nanotron.distributed import dist
from nanotron.helpers import _vocab_size_with_padding
from nanotron.models import build_model
from mamba import MambaForTraining
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import NanotronParameter, sanity_check
from nanotron.serialize import save_meta, save_weights
from nanotron.trainer import mark_tied_parameters


def get_weight_from_hf(
    name: str,
    ref_module_state_dict: Dict[str, torch.Tensor],
    ref_module: MambaLMHeadModel,
    nanotron_to_hf: Dict[str, str],
    get_grad: bool = False
) -> torch.Tensor:
    """From our brrr implementation, we get the equivalent tensor in transformers implementation"""

    hf_name = nanotron_to_hf[name]

    if get_grad is False:

        def get_tensor(path: str):
            return ref_module_state_dict[path]
    else:

        def get_tensor(path: str):
            weight = ref_module.get_parameter(path)
            return weight.grad

    return get_tensor(hf_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HF weights from states-space repo to brrr weights")
    parser.add_argument("--model", type=str, default="130M", help="130M | 370M | 790M | 1.4B | 2.8B")
    parser.add_argument("--save_path", type=str, default="mamba-nanotron")
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    args = parser.parse_args()

    if args.model not in ["130M", "370M", "790M", "1.4B", "2.8B"]:
        raise ValueError("Model should be one of 130M, 370M, 790M, 1.4B, 2.8B")

    if args.tp > 1:
        raise ValueError("Tensor parallelism not supported yet (as A_log is nn.Parameter which is not a NanotronParameter and thus cannot be marked as is_sharded)")

    save_path = Path(args.save_path)

    #TODO: Do it this way so that we can choose the dp pp and tp we want
    # https://github.com/huggingface/brrr/blob/main/legacy_examples/starcoder2/convert_brrr_to_trfrs.py
    parallel_config = ParallelismArgs(
        dp=args.dp,
        pp=args.pp,
        tp=args.tp,
        pp_engine=AllForwardAllBackwardPipelineEngine(),
        tp_mode=TensorParallelLinearMode.ALL_REDUCE,
        tp_linear_async_communication=False,
    )

    parallel_context = ParallelContext(
        data_parallel_size=parallel_config.dp,
        pipeline_parallel_size=parallel_config.pp,
        tensor_parallel_size=parallel_config.tp,
    )

    d_model = None
    num_hidden_layers = None
    pretrained_model_name = None

    if args.model == "130M":
        d_model = 768
        num_hidden_layers = 24
        pretrained_model_name = "state-spaces/mamba-130m"
    elif args.model == "370M":
        d_model = 1024
        num_hidden_layers = 48
        pretrained_model_name = "state-spaces/mamba-370m"
    elif args.model == "790M":
        d_model = 1536
        num_hidden_layers = 24
        pretrained_model_name = "state-spaces/mamba-790m"
    elif args.model == "1.4B":
        d_model = 2048
        num_hidden_layers = 48
        pretrained_model_name = "state-spaces/mamba-1.4b"
    elif args.model == "2.8B":
        d_model = 2560
        num_hidden_layers = 64
        pretrained_model_name = "state-spaces/mamba-2.8b"

    yaml_content = f"""
    is_mamba_config: true
    d_model: {d_model}
    dtype: bfloat16
    fused_add_norm: true
    is_mamba_config: true
    num_hidden_layers: {num_hidden_layers}
    pad_token_id: null
    pad_vocab_size_multiple: 8
    residual_in_fp32: true
    rms_norm: true
    rms_norm_eps: 1.0e-05
    ssm_cfg: null
    vocab_size: 50277
    """

    str_to_dtype = {
        "float32": torch.float32,
        "float64": torch.float64,
        "complex64": torch.complex64,
        "complex128": torch.complex128,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "uint8": torch.uint8,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "bool": torch.bool,
    }

    attrs = yaml.safe_load(yaml_content)
    model_config = MambaModelConfig(**attrs)

    # # Initiliaze Brrr model
    # model_config.vocab_size = _vocab_size_with_padding(
    #         model_config.vocab_size,
    #         pg_size=parallel_context.tp_pg.size(),
    #         make_vocab_size_divisible_by=1,
    # )

    nanotron_model = build_model(
        model_builder=lambda: MambaForTraining(
            config=model_config,
            parallel_context=parallel_context,
            parallel_config=parallel_config,
            random_states=None,
        ),
        parallel_context=parallel_context,
        dtype=str_to_dtype[model_config.dtype],
        device=torch.device("cpu")
    )

    device_map = {}
    current_pp_rank = dist.get_rank(parallel_context.pp_pg)

    tied_embs_ranks = [nanotron_model.model.token_position_embeddings.rank, nanotron_model.model.lm_head.rank]

    device_map["backbone.embedding"] = (
        nanotron_model.model.token_position_embeddings.rank if current_pp_rank in tied_embs_ranks else "meta"
    )

    for i in range(model_config.num_hidden_layers):
        device_map[f"backbone.layers[{i}]"] = (
            nanotron_model.model.decoder[i].rank if current_pp_rank == nanotron_model.model.decoder[i].rank else "meta"
        )

    device_map["lm_head"] = nanotron_model.model.lm_head.rank if current_pp_rank in tied_embs_ranks else "meta"

    model_ref = MambaLMHeadModel.from_pretrained(pretrained_model_name, device="cpu", dtype=str_to_dtype[model_config.dtype])

    # Create a mapping from nanotron to hf
    nanotron_to_hf = {}

    for i in range(model_config.num_hidden_layers):
        nanotron_to_hf[f'decoder.{i}.pp_block.mixer.A_log'] = f'backbone.layers.{i}.mixer.A_log'
        nanotron_to_hf[f'decoder.{i}.pp_block.mixer.D'] = f'backbone.layers.{i}.mixer.D'
        nanotron_to_hf[f'decoder.{i}.pp_block.mixer.in_proj.weight'] = f'backbone.layers.{i}.mixer.in_proj.weight'
        nanotron_to_hf[f'decoder.{i}.pp_block.mixer.conv1d.weight'] = f'backbone.layers.{i}.mixer.conv1d.weight'
        nanotron_to_hf[f'decoder.{i}.pp_block.mixer.conv1d.bias'] = f'backbone.layers.{i}.mixer.conv1d.bias'
        nanotron_to_hf[f'decoder.{i}.pp_block.mixer.x_proj.weight'] = f'backbone.layers.{i}.mixer.x_proj.weight'
        nanotron_to_hf[f'decoder.{i}.pp_block.mixer.x_proj.bias'] = f'backbone.layers.{i}.mixer.x_proj.bias'
        nanotron_to_hf[f'decoder.{i}.pp_block.mixer.dt_proj.weight'] = f'backbone.layers.{i}.mixer.dt_proj.weight'
        nanotron_to_hf[f'decoder.{i}.pp_block.mixer.dt_proj.bias'] = f'backbone.layers.{i}.mixer.dt_proj.bias'
        nanotron_to_hf[f'decoder.{i}.pp_block.mixer.out_proj.weight'] = f'backbone.layers.{i}.mixer.out_proj.weight'
        #TODO: Maybe check if bias exists?
        nanotron_to_hf[f'decoder.{i}.pp_block.mixer.out_proj.bias'] = f'backbone.layers.{i}.mixer.out_proj.bias'
        nanotron_to_hf[f'decoder.{i}.pp_block.norm.weight'] = f'backbone.layers.{i}.norm.weight'

    nanotron_to_hf['token_position_embeddings.pp_block.token_embedding.weight'] = 'backbone.embedding.weight'
    nanotron_to_hf['final_layer_norm.pp_block.weight'] = 'backbone.norm_f.weight'
    nanotron_to_hf['lm_head.pp_block.weight'] = 'lm_head.weight'

    # Sync weights
    ref_state_dict = model_ref.state_dict()
    for name, param in tqdm(nanotron_model.model.named_parameters(), total=len(list(nanotron_model.model.named_parameters())), desc="Converting"):
        ref_param = get_weight_from_hf(name=name, ref_module_state_dict=ref_state_dict, ref_module=model_ref, nanotron_to_hf=nanotron_to_hf)

        param_is_tp_sharded = (
            isinstance(param, NanotronParameter)
            and param.is_sharded
            and parallel_context.world_ranks_to_pg[param.get_sharded_info().global_ranks] == parallel_context.tp_pg
        )

        if param_is_tp_sharded:
            sharded_info = param.get_sharded_info()
            # copy param data (not just the reference)
            with torch.no_grad():
                for local_global_slices_pair in sharded_info.local_global_slices_pairs:
                    local_slices = local_global_slices_pair.local_slices
                    global_slices = local_global_slices_pair.global_slices
                    param[local_slices].copy_(ref_param[global_slices])
        else:
            assert (
                ref_param.shape == param.shape
            ), f"Parameter shape don't match for {name}\n{ref_param.shape} != {param.shape}"
            # copy param data (not just the reference)
            with torch.no_grad():
                param.copy_(ref_param)
                ref_param = None
                torch.cuda.empty_cache()

    # Marks parameters as NanotronParameters
    mark_tied_parameters(model=nanotron_model, parallel_context=parallel_context)

    sanity_check(root_module=nanotron_model)
    save_weights(model=nanotron_model, parallel_context=parallel_context, root_folder=save_path)
    checkpoint_metadata = {
        "last_train_step": 0,
        "consumed_train_samples": 0,
    }
    save_meta(root_folder=save_path, parallel_context=parallel_context, checkpoint_metadata=checkpoint_metadata)

    with open(save_path / "config.yaml", "w") as f:
        config = MambaConfig(
            general=GeneralArgs(project="test", run="mamba"),
            parallelism=parallel_config,
            model=ModelArgs(
                    init_method=MambaInit(),
                    model_config=model_config,
                ),
            tokenizer=TokenizerArgs("EleutherAI/gpt-neox-20b"),
        )
        print("Saving config ...")
        yaml.dump(config.as_dict(), f)