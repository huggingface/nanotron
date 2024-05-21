from typing import Dict, List, Tuple, Union

import torch
import torch.distributed as dist
from nanotron.models.base import NanotronModel
from nanotron.parallel import ParallelContext
from torch import nn
from torch.distributed import ReduceOp


def track_weight_and_grad_stats(name: str, module: nn.Module, parallel_context: ParallelContext):
    def compute_stas(tensors):
        tensors = {"tensor": tensors} if not isinstance(tensors, dict) else tensors

        # if isinstance(tensor, dict):
        #     assert 1 == 1

        stats = {}
        for key, tensor in tensors.items():
            stats[key] = {}
            stats[key] = {
                # "mean": tensor.mean().item(),
                # "std": tensor.std().item(),
                # "var": tensor.var().item(),
                # "norm": tensor.norm().item(),
                # "min": tensor.min().item(),
                # "max": tensor.max().item(),
                "amax": tensor.abs()
                .max()
                .item(),
            }

            # NOTE: now all reduce mean this across tp ranks
            tp_group = parallel_context.tp_pg
            for metric_name, metric_value in stats[key].items():

                stats[key][metric_name] = torch.tensor(metric_value, device=tensor.device, dtype=tensor.dtype)

                dist.all_reduce(stats[key][metric_name], op=ReduceOp.MAX, group=tp_group)

                # if stats[key][metric_name].is_floating_point():
                #     stats[key][metric_name] /= parallel_context.tensor_parallel_size

        return stats

    logs: Dict[str, Dict[str, float]] = {}

    if name not in logs:
        logs[name] = {}

    def _save_output_stats(module: nn.Linear, input: torch.Tensor, output: torch.Tensor):
        if hasattr(module, "weight") and module.weight is not None:
            logs[name]["weight"] = compute_stas(module.weight.data)
            # logging[name]["weight"] = _collect_stats(module.weight)

        if hasattr(module, "bias") and module.bias is not None:
            logs[name]["bias"] = compute_stas(module.bias)

        inputs = input if isinstance(input, tuple) else (input,)
        outputs = output if isinstance(output, tuple) else (output,)

        if len(inputs) > 1:
            for i, inp in enumerate(inputs):
                if inp.dtype == torch.long:
                    # NOTE: this is input ids in transformers
                    continue
                logs[name][f"input:{i}"] = compute_stas(inp)
        elif len(inputs) == 1:
            logs[name]["input"] = compute_stas(inputs[0])

        if len(outputs) > 1:
            for i, out in enumerate(outputs):
                logs[name][f"output:{i}"] = compute_stas(out)
        elif len(outputs) == 1:
            logs[name]["output"] = compute_stas(outputs[0])

    def _save_grad_stats(module: nn.Linear, grad_input, grad_output: torch.Tensor):
        # import pydevd
        # pydevd.settrace(suspend=False, trace_only_current_thread=True)
        # logging[name][f"weight_grad"] = _collect_stats(module.weight.grad.orig_data)
        # logging[name][f"bias_grad"] = _collect_stats(module.bias.grad)

        if isinstance(grad_output, tuple):
            for i, grad in enumerate(grad_output):
                if grad is None:
                    continue

                logs[name][f"grad_output:{i}"] = compute_stas(grad)
        else:
            logs[name]["grad_output"] = compute_stas(grad_output)

        if isinstance(grad_input, tuple):
            for i, grad in enumerate(grad_input):
                if grad is not None:
                    logs[name][f"grad_input:{i}"] = compute_stas(grad)
        else:
            if grad_input is not None:
                logs[name]["grad_input"] = compute_stas(grad_input)

    handles = []
    handles.append(module.register_forward_hook(_save_output_stats))
    # module.register_full_backward_pre_hook(_save_grad_stats)
    handles.append(module.register_backward_hook(_save_grad_stats))
    return logs, handles
    # module.register_module_full_backward_hook(_save_grad_stats)


def monitor_nanotron_model(model: NanotronModel, parallel_context: ParallelContext):
    def get_leaf_modules(module: nn.Module) -> List[Tuple[str, nn.Module]]:
        """
        Return all the leaf modules (modules without any child modules) in a PyTorch module.
        """
        leaf_modules = []
        for n, m in module.named_modules():
            if not list(m.children()):
                leaf_modules.append((n, m))
        return leaf_modules

    logs = {}
    handles = []
    leaf_modules = get_leaf_modules(model)

    for name, module in leaf_modules:
        module_logs, module_handles = track_weight_and_grad_stats(name, module, parallel_context)
        logs.update(module_logs)
        handles.extend(module_handles)

    return logs, handles


def convert_logs_to_flat_logs(
    logs: Dict[str, Dict[str, Dict[str, Union[torch.Tensor, float]]]]
) -> Dict[str, Union[torch.Tensor, float]]:
    flat_logs = {}
    for module_name, components in logs.items():
        for component_name, stats in components.items():
            for metric_name, metric_value in stats.items():
                flat_logs[f"{module_name}:{component_name}:{metric_name}"] = metric_value

    return flat_logs
