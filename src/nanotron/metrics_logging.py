"""
Utility functions for computing and logging various model metrics.
"""

from typing import Any, Dict, Optional

import torch

########################
# Basic utility functions
########################


def compute_tensor_norm(tensor: torch.Tensor, p: int = 2) -> torch.Tensor:
    """Compute the Lp norm of a tensor."""
    return torch.linalg.vector_norm(tensor, ord=p)


def get_attribute_by_path(obj: Any, path: str) -> Optional[Any]:
    """Get an attribute from an object by a dot-separated path."""
    path_parts = path.split(".")
    current = obj

    for i, attr in enumerate(path_parts):
        if hasattr(current, attr):
            current = getattr(current, attr)
        else:
            return None

    if isinstance(current, torch.Tensor) and hasattr(current, "detach"):
        return current.detach()

    return current


########################
# Statistical functions
########################


def compute_kurtosis(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute excess kurtosis which measures "tailedness" of distribution.
    High kurtosis indicates outliers, low kurtosis indicates uniform-like distribution.
    """
    device = tensor.device
    tensor = tensor.float()

    mean = tensor.mean()
    std = tensor.std()
    if std == 0:
        return torch.tensor(0.0, device=device)

    normalized = (tensor - mean) / std
    return torch.mean(normalized.pow(4)) - 3  # Excess kurtosis (normal distribution has kurtosis=3)


def compute_zero_fraction(tensor: torch.Tensor, threshold: float = 1e-10) -> torch.Tensor:
    """Compute fraction of near-zero elements."""
    return (tensor.abs() < threshold).float().mean()


def compute_tensor_stats(tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Compute comprehensive statistics for a tensor."""
    if tensor.numel() == 0:
        device = tensor.device
        return {
            "mean": torch.tensor(0.0, device=device),
            "std": torch.tensor(0.0, device=device),
            "min": torch.tensor(0.0, device=device),
            "max": torch.tensor(0.0, device=device),
            "norm_l1": torch.tensor(0.0, device=device),
            "norm_l2": torch.tensor(0.0, device=device),
            "zero_frac": torch.tensor(1.0, device=device),
            "kurtosis": torch.tensor(0.0, device=device),
        }

    tensor_f = tensor.float()
    mean = tensor_f.mean()
    std = torch.sqrt(tensor_f.var())

    return {
        "mean": mean,
        "std": std,
        "min": tensor_f.min(),
        "max": tensor_f.max(),
        "norm_l1": compute_tensor_norm(tensor_f, p=1),
        "norm_l2": compute_tensor_norm(tensor_f, p=2),
        "zero_frac": compute_zero_fraction(tensor_f),
        "kurtosis": compute_kurtosis(tensor_f),
    }


class MetricsLogger:
    """
    Class for logging experiment metrics with configurable detail levels.

    Supports two modes:
    - Basic (level=0): Only logs essential metrics (loss, accuracy, performance)
    - Full (level=1): Logs all metrics including detailed model statistics
    """

    # Standard component paths for finding model weights
    MODEL_COMPONENTS = {
        "attention": {
            "qkv_proj": "model.decoder.{}.pp_block.attn.qkv_proj.weight",
            "o_proj": "model.decoder.{}.pp_block.attn.o_proj.weight",
        },
        "mlp": {
            "gate_up_proj": "model.decoder.{}.pp_block.mlp.gate_up_proj.weight",
            "down_proj": "model.decoder.{}.pp_block.mlp.down_proj.weight",
        },
        "norm": {
            "input_layernorm": "model.decoder.{}.pp_block.input_layernorm.weight",
            "post_attention_layernorm": "model.decoder.{}.pp_block.post_attention_layernorm.weight",
        },
    }

    EMBEDDING_PATHS = [
        "model.token_position_embeddings.pp_block.token_embedding.weight",
        "model.lm_head.pp_block.weight",
    ]

    def __init__(self, config):
        """Initialize the logger with configuration."""
        self.config = config
        if self.config.metrics_logging is not None:
            self.log_level = config.metrics_logging.log_level
            self.log_detail_interval = config.metrics_logging.log_detail_interval
        else:
            self.log_level = 0
            self.log_detail_interval = 1

    def _format_paths(self, components: Dict, max_layers: int) -> Dict:
        """Pre-format component paths with layer indices for efficiency."""
        formatted = {}
        for comp_type, subcomponents in components.items():
            formatted[comp_type] = {}
            for subcomp_name, pattern in subcomponents.items():
                formatted[comp_type][subcomp_name] = [pattern.format(i) for i in range(max_layers)]
        return formatted

    def collect_embeddings_metrics(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """Collect metrics for model embeddings."""
        metrics = {}

        for path in self.EMBEDDING_PATHS:
            embeddings = get_attribute_by_path(model, path)
            if embeddings is not None:
                name = path.split(".")[-2]
                stats = compute_tensor_stats(embeddings)

                for stat_name, value in stats.items():
                    metrics[f"{name}/{stat_name}"] = value

                metrics[f"{name}/size"] = torch.tensor(embeddings.numel(), device=embeddings.device)

        return metrics

    def compute_global_hidden_layer_metrics(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """
        Compute global metrics across all hidden layers, excluding embeddings, final layernorm, and lm_head.
        Aggregates weights from all decoder layers to provide model-wide statistics.
        """
        metrics = {}
        max_layers = self.config.model.model_config.num_hidden_layers

        if max_layers == 0:
            return metrics

        formatted_paths = self._format_paths(self.MODEL_COMPONENTS, max_layers)

        # Group weights by component type
        component_weights = {comp_type: [] for comp_type in self.MODEL_COMPONENTS}
        all_layer_weights = []

        # Collect all weights from hidden layers
        for layer_idx in range(max_layers):
            for comp_type, subcomponents in self.MODEL_COMPONENTS.items():
                for subcomp_name in subcomponents:
                    path = formatted_paths[comp_type][subcomp_name][layer_idx]
                    param = get_attribute_by_path(model, path)
                    if param is not None:
                        param_tensor = param.detach().float()
                        all_layer_weights.append(param_tensor)
                        component_weights[comp_type].append(param_tensor)

        if not all_layer_weights:
            return metrics

        # Compute statistics for each component type
        for comp_type, weights in component_weights.items():
            if not weights:
                continue

            # Flatten tensors for global statistics calculation
            flat_tensors = [w.reshape(-1) for w in weights]
            all_comp_weights = torch.cat(flat_tensors)
            prefix = f"global_{comp_type}"

            comp_stats = compute_tensor_stats(all_comp_weights)
            for stat_name, value in comp_stats.items():
                metrics[f"{prefix}/{stat_name}"] = value

        # Compute global stats across all hidden layers
        flat_all_tensors = [w.reshape(-1) for w in all_layer_weights]
        all_weights = torch.cat(flat_all_tensors)

        global_stats = compute_tensor_stats(all_weights)
        for stat_name, value in global_stats.items():
            metrics[f"global_global/{stat_name}"] = value

        return metrics

    def collect_parameter_metrics(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """Collect detailed metrics for model parameters by layer and component."""
        metrics = {}
        max_layers = self.config.model.model_config.num_hidden_layers

        formatted_paths = self._format_paths(self.MODEL_COMPONENTS, max_layers)

        # Collect metrics for each layer
        for layer_idx in range(max_layers):
            layer_name = f"layer_{layer_idx}"

            for comp_type, subcomponents in self.MODEL_COMPONENTS.items():
                for subcomp_name in subcomponents:
                    path = formatted_paths[comp_type][subcomp_name][layer_idx]
                    param = get_attribute_by_path(model, path)

                    if param is not None:
                        # Add parameter metrics
                        stats = compute_tensor_stats(param)
                        for stat_name, value in stats.items():
                            metrics[f"{layer_name}/{comp_type}/{subcomp_name}/{stat_name}"] = value

                        # Add gradient stats if available
                        if hasattr(param, "grad") and param.grad is not None:
                            grad_stats = compute_tensor_stats(param.grad.detach())
                            for stat_name, value in grad_stats.items():
                                metrics[f"{layer_name}/{comp_type}/{subcomp_name}/grad/{stat_name}"] = value

        # Get final layer norm
        final_ln = get_attribute_by_path(model, "model.final_layer_norm.pp_block.weight")
        if final_ln is not None:
            stats = compute_tensor_stats(final_ln)
            for stat_name, value in stats.items():
                metrics[f"final_layernorm/{stat_name}"] = value

        return metrics

    def collect_all_metrics(
        self,
        model: torch.nn.Module,
    ) -> Dict[str, torch.Tensor]:
        """Collect all metrics based on the specified log level and iteration."""
        metrics = {}
        metrics.update(self.compute_global_hidden_layer_metrics(model))
        metrics.update(self.collect_embeddings_metrics(model))
        metrics.update(self.collect_parameter_metrics(model))
        return metrics


if __name__ == "__main__":
    # Simple test code
    print("Metrics logging utilities loaded")
