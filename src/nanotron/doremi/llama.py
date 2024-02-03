from typing import Dict, Optional, Union

import torch
from nanotron import logging
from nanotron.config import ParallelismArgs
from nanotron.doremi.doremi_context import DoReMiContext
from nanotron.doremi.loss import CrossEntropyWithPerDomainLoss, DoReMiLossForProxyTraining
from nanotron.models import NanotronModel
from nanotron.models.fast.llama import LlamaModel
from nanotron.nn.layer_norm import TritonRMSNorm
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import NanotronParameter
from nanotron.parallel.pipeline_parallel.block import PipelineBlock, TensorPointer
from nanotron.parallel.tensor_parallel.functional import sharded_cross_entropy
from nanotron.parallel.tensor_parallel.nn import (
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    TensorParallelRowLinear,
)
from transformers import LlamaConfig

logger = logging.get_logger(__name__)


class BaseLLaMa(NanotronModel):
    @torch.no_grad()
    def init_model_randomly(self, init_method, scaled_init_method):
        """Initialize model parameters randomly.
        Args:
            init_method (callable): Used for embedding/position/qkv weight in attention/first layer weight of mlp/ /lm_head/
            scaled_init_method (callable): Used for o weight in attention/second layer weight of mlp/

        Note:
            Layernorm weight all 0 or 1 depending on `apply_layernorm_1p`
        """
        model = self
        initialized_parameters = set()
        # Handle tensor parallelism
        module_id_to_prefix = {id(module): f"{module_name}." for module_name, module in model.named_modules()}
        # Fix the root_model
        module_id_to_prefix[id(model)] = ""

        for module_name, module in model.named_modules():
            if isinstance(module, TensorParallelColumnLinear):
                # Somehow Megatron-LM does something super complicated, https://github.com/NVIDIA/Megatron-LM/blob/2360d732a399dd818d40cbe32828f65b260dee11/megatron/core/tensor_parallel/layers.py#L96
                # What it does:
                #  - instantiate a buffer of the `full size` in fp32
                #  - run init method on it
                #  - shard result to get only a specific shard
                # Instead I'm lazy and just going to run init_method, since they are scalar independent
                assert {"weight"} == {name for name, _ in module.named_parameters()} or {"weight"} == {
                    name for name, _ in module.named_parameters()
                }
                for param_name, param in module.named_parameters():
                    assert isinstance(param, NanotronParameter)
                    if param.is_tied:
                        tied_info = param.get_tied_info()
                        full_param_name = tied_info.get_full_name_from_module_id_to_prefix(
                            module_id_to_prefix=module_id_to_prefix
                        )
                    else:
                        full_param_name = f"{module_name}.{param_name}"

                    if full_param_name in initialized_parameters:
                        # Already initialized
                        continue

                    if "weight" == param_name:
                        init_method(param)
                    elif "bias" == param_name:
                        param.zero_()
                    else:
                        raise ValueError(f"Who the fuck is {param_name}?")

                    assert full_param_name not in initialized_parameters
                    initialized_parameters.add(full_param_name)
            elif isinstance(module, TensorParallelRowLinear):
                # Somehow Megatron-LM does something super complicated, https://github.com/NVIDIA/Megatron-LM/blob/2360d732a399dd818d40cbe32828f65b260dee11/megatron/core/tensor_parallel/layers.py#L96
                # What it does:
                #  - instantiate a buffer of the `full size` in fp32
                #  - run init method on it
                #  - shard result to get only a specific shard
                # Instead I'm lazy and just going to run init_method, since they are scalar independent
                assert {"weight"} == {name for name, _ in module.named_parameters()} or {"weight"} == {
                    name for name, _ in module.named_parameters()
                }
                for param_name, param in module.named_parameters():
                    assert isinstance(param, NanotronParameter)
                    if param.is_tied:
                        tied_info = param.get_tied_info()
                        full_param_name = tied_info.get_full_name_from_module_id_to_prefix(
                            module_id_to_prefix=module_id_to_prefix
                        )
                    else:
                        full_param_name = f"{module_name}.{param_name}"

                    if full_param_name in initialized_parameters:
                        # Already initialized
                        continue

                    if "weight" == param_name:
                        scaled_init_method(param)
                    elif "bias" == param_name:
                        param.zero_()
                    else:
                        raise ValueError(f"Who the fuck is {param_name}?")

                    assert full_param_name not in initialized_parameters
                    initialized_parameters.add(full_param_name)
            elif isinstance(module, TritonRMSNorm):
                assert {"weight"} == {name for name, _ in module.named_parameters()}
                for param_name, param in module.named_parameters():
                    assert isinstance(param, NanotronParameter)
                    if param.is_tied:
                        tied_info = param.get_tied_info()
                        full_param_name = tied_info.get_full_name_from_module_id_to_prefix(
                            module_id_to_prefix=module_id_to_prefix
                        )
                    else:
                        full_param_name = f"{module_name}.{param_name}"

                    if full_param_name in initialized_parameters:
                        # Already initialized
                        continue

                    if "weight" == param_name:
                        # TODO @thomasw21: Sometimes we actually want 0
                        param.fill_(1)
                    elif "bias" == param_name:
                        param.zero_()
                    else:
                        raise ValueError(f"Who the fuck is {param_name}?")

                    assert full_param_name not in initialized_parameters
                    initialized_parameters.add(full_param_name)
            elif isinstance(module, TensorParallelEmbedding):
                # TODO @thomasw21: Handle tied embeddings
                # Somehow Megatron-LM does something super complicated, https://github.com/NVIDIA/Megatron-LM/blob/2360d732a399dd818d40cbe32828f65b260dee11/megatron/core/tensor_parallel/layers.py#L96
                # What it does:
                #  - instantiate a buffer of the `full size` in fp32
                #  - run init method on it
                #  - shard result to get only a specific shard
                # Instead I'm lazy and just going to run init_method, since they are scalar independent
                assert {"weight"} == {name for name, _ in module.named_parameters()}

                assert isinstance(module.weight, NanotronParameter)
                if module.weight.is_tied:
                    tied_info = module.weight.get_tied_info()
                    full_param_name = tied_info.get_full_name_from_module_id_to_prefix(
                        module_id_to_prefix=module_id_to_prefix
                    )
                else:
                    full_param_name = f"{module_name}.weight"

                if full_param_name in initialized_parameters:
                    # Already initialized
                    continue

                init_method(module.weight)
                assert full_param_name not in initialized_parameters
                initialized_parameters.add(full_param_name)

        assert initialized_parameters == {
            param.get_tied_info().get_full_name_from_module_id_to_prefix(module_id_to_prefix=module_id_to_prefix)
            if param.is_tied
            else name
            for name, param in model.named_parameters()
        }, f"Somehow the initialized set of parameters don't match:\n - Expected: { {name for name, _ in model.named_parameters()} }\n - Got: {initialized_parameters}"

    def get_block_compute_costs(self):
        """Computes the compute cost of each block in the model so that we can do a better job of load balancing."""
        return self.model.get_block_compute_costs()

    def get_flops_per_sec(self, iteration_time_in_sec, sequence_length, global_batch_size):
        """Get flops per second for a given model"""
        return self.model.get_flops_per_sec(iteration_time_in_sec, sequence_length, global_batch_size)


class LLaMaForInference(BaseLLaMa):
    def __init__(
        self,
        config: LlamaConfig,
        parallel_config: Optional[ParallelismArgs],
        parallel_context: ParallelContext,
    ):
        super().__init__()
        self.model = LlamaModel(config=config, parallel_context=parallel_context, parallel_config=parallel_config)
        self.parallel_context = parallel_context
        self.config = config
        self.parallel_config = parallel_config

    def forward(
        self,
        input_ids: Union[torch.Tensor, TensorPointer],
        input_mask: Union[torch.Tensor, TensorPointer],
        label_ids: Union[torch.Tensor, TensorPointer],
        label_mask: Union[torch.Tensor, TensorPointer],
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        sharded_logits = self.model(
            input_ids=input_ids,
            input_mask=input_mask,
        )
        loss = sharded_cross_entropy(
            sharded_logits,
            label_ids.transpose(0, 1).contiguous(),
            group=self.parallel_context.tp_pg,
            dtype=torch.float,
        ).transpose(0, 1)
        # per_token_losses = loss * label_mask
        return {"losses": loss}


class LlamaForDoReMiTraining(BaseLLaMa):
    def __init__(
        self,
        config: LlamaConfig,
        parallel_context: ParallelContext,
        doremi_context: DoReMiContext,
        parallel_config: Optional[ParallelismArgs],
    ):
        super().__init__()
        self.model = LlamaModel(config=config, parallel_context=parallel_context, parallel_config=parallel_config)
        self.loss = PipelineBlock(
            p2p=self.model.p2p,
            module_builder=DoReMiLossForProxyTraining,
            module_kwargs={
                "parallel_context": parallel_context,
                "doremi_context": doremi_context,
            },
            module_input_keys={
                "sharded_logits",
                "label_ids",
                "label_mask",
                "domain_idxs",
                "ref_losses",
            },
            module_output_keys={"loss", "excess_losses", "domain_losses", "domain_weights", "samples_per_domain"},
        )
        self.parallel_context = parallel_context
        self.config = config
        self.parallel_config = parallel_config

    def forward(
        self,
        input_ids: Union[torch.Tensor, TensorPointer],
        input_mask: Union[torch.Tensor, TensorPointer],
        label_ids: Union[torch.Tensor, TensorPointer],
        label_mask: Union[torch.Tensor, TensorPointer],
        domain_idxs: Optional[Union[torch.Tensor, TensorPointer]],
        ref_losses: Optional[Union[torch.Tensor, TensorPointer]],
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        sharded_logits = self.model(
            input_ids=input_ids,
            input_mask=input_mask,
        )
        sharded_logits = sharded_logits.transpose(0, 1).contiguous()
        # label_ids = label_ids.transpose(0, 1).contiguous()
        outputs = self.loss(
            sharded_logits=sharded_logits,
            label_ids=label_ids,
            label_mask=label_mask,
            domain_idxs=domain_idxs,
            ref_losses=ref_losses,
        )
        return outputs


class LlamaReferenceForTrainingWithPerDomainLoss(BaseLLaMa):
    def __init__(
        self,
        config: LlamaConfig,
        doremi_context: DoReMiContext,
        parallel_context: ParallelContext,
        parallel_config: Optional[ParallelismArgs],
    ):
        super().__init__()
        self.model = LlamaModel(config=config, parallel_context=parallel_context, parallel_config=parallel_config)
        self.loss = PipelineBlock(
            p2p=self.model.p2p,
            module_builder=CrossEntropyWithPerDomainLoss,
            module_kwargs={
                "doremi_context": doremi_context,
                "parallel_context": parallel_context,
            },
            module_input_keys={"sharded_logits", "label_ids", "label_mask", "domain_idxs"},
            module_output_keys={"loss", "domain_losses", "samples_per_domain"},
        )
        self.parallel_context = parallel_context
        self.config = config
        self.parallel_config = parallel_config

    def forward(
        self,
        input_ids: Union[torch.Tensor, TensorPointer],
        input_mask: Union[torch.Tensor, TensorPointer],
        label_ids: Union[torch.Tensor, TensorPointer],
        label_mask: Union[torch.Tensor, TensorPointer],
        domain_idxs: Union[torch.Tensor, TensorPointer],
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        sharded_logits = self.model(
            input_ids=input_ids,
            input_mask=input_mask,
        )
        sharded_logits = sharded_logits.transpose(0, 1).contiguous()
        outputs = self.loss(
            sharded_logits=sharded_logits,
            label_ids=label_ids,
            label_mask=label_mask,
            domain_idxs=domain_idxs,
        )
        return {
            "loss": outputs["loss"],
            "domain_losses": outputs["domain_losses"],
            "samples_per_domain": outputs["samples_per_domain"],
        }
