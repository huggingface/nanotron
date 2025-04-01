"""
Utility functions for computing and logging various model metrics.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

########################
# Basic utility functions
########################


def compute_tensor_norm(tensor: torch.Tensor, p: int = 2) -> torch.Tensor:
    """Compute the Lp norm of a tensor."""
    return torch.linalg.vector_norm(tensor, ord=p)


def get_attribute_by_path(obj: Any, path: str, debug: bool = False) -> Optional[Any]:
    """Get an attribute from an object by a dot-separated path."""
    path_parts = path.split(".")
    current = obj
    
    for i, attr in enumerate(path_parts):
        if hasattr(current, attr):
            current = getattr(current, attr)
        else:
            if debug:
                print(f"Could not find attribute '{attr}' in path '{path}' at position {i}")
            return None

    if isinstance(current, torch.Tensor) and hasattr(current, "detach"):
        return current.detach()

    return current


########################
# Statistical functions
########################


def compute_kurtosis(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute kurtosis which measures "tailedness" of distribution.
    High kurtosis indicates outliers, low kurtosis indicates uniform-like distribution.
    """
    # Use float() directly on tensor operations rather than making a copy
    mean = tensor.float().mean()
    std = tensor.float().std()
    # Avoid division by zero
    if std == 0:
        return torch.tensor(0.0, device=tensor.device)

    # Compute kurtosis: E[((x-μ)/σ)^4]
    # Operate on the tensor directly without flattening
    normalized = ((tensor.float() - mean) / std)
    # Using .pow(4) is more efficient than **4
    return torch.mean(normalized.pow(4)) - 3  # Excess kurtosis (normal distribution has kurtosis=3)


def compute_zero_fraction(tensor: torch.Tensor, threshold: float = 1e-10) -> torch.Tensor:
    """Compute fraction of near-zero elements."""
    return (tensor.abs() < threshold).float().mean()


def compute_activation_stats(tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Compute comprehensive statistics for a tensor."""
    if tensor.numel() == 0:
        # Create empty stats with consistent device
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

    # Convert to float once and reuse
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


class ExperimentLogger:
    """
    Class for logging experiment metrics with configurable detail levels.
    
    Supports two modes:
    - Basic (level=0): Only logs essential metrics (loss, accuracy, performance)
    - Full (level=1): Logs all metrics including detailed model statistics
    """
    
    # Standard component patterns used by multiple methods
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
        """
        Initialize the logger with configuration.
        
        Args:
            config: Configuration object with metrics_logging attribute
        """
        self.config = config
        self.log_level = config.metrics_logging.level
        self.log_interval = config.metrics_logging.interval
    
    def should_log_detailed_metrics(self, iteration: int) -> bool:
        """
        Determine if detailed metrics should be logged for the current iteration.
        
        Args:
            iteration: Current training iteration/step
            
        Returns:
            bool: Whether to log detailed metrics
        """
        # Always log first few iterations for debugging
        if iteration < 5:
            return True
            
        # Log according to interval if log_level is set to full (1)
        if self.log_level > 0 and iteration % self.log_interval == 0:
            return True
            
        return False
    
    def _format_paths(self, components: Dict, max_layers: int) -> Dict:
        """
        Pre-format component paths with layer indices to avoid repeated formatting.
        
        Args:
            components: Dict of component patterns
            max_layers: Number of layers to format
            
        Returns:
            Dict of pre-formatted paths
        """
        formatted = {}
        for comp_type, subcomponents in components.items():
            formatted[comp_type] = {}
            for subcomp_name, pattern in subcomponents.items():
                formatted[comp_type][subcomp_name] = [
                    pattern.format(i) for i in range(max_layers)
                ]
        return formatted
    
    def collect_embeddings_metrics(self, model: torch.nn.Module, debug: bool = False) -> Dict[str, torch.Tensor]:
        """Collect metrics for model embeddings."""
        metrics = {}

        for path in self.EMBEDDING_PATHS:
            embeddings = get_attribute_by_path(model, path, debug=debug)
            if embeddings is not None:
                name = path.split(".")[-2]
                stats = compute_activation_stats(embeddings)
                
                for stat_name, value in stats.items():
                    metrics[f"{name}/{stat_name}"] = value
                
                metrics[f"{name}/size"] = torch.tensor(embeddings.numel(), device=embeddings.device)

        return metrics
    
    def compute_global_hidden_layer_metrics(self, model: torch.nn.Module, debug: bool = False) -> Dict[str, torch.Tensor]:
        """
        Compute global metrics across all hidden layers, excluding embeddings, final layernorm, and lm_head.
        Aggregates weights from all decoder layers to provide model-wide statistics.
        """
        metrics = {}
        
        # Get number of layers directly from config
        max_layers = self.config.model.model_config.num_hidden_layers
        
        if max_layers == 0:
            return metrics
        
        # Pre-format the paths to avoid repeated formatting in loops
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
                
            flat_tensors = [w.reshape(-1) for w in weights]
            all_comp_weights = torch.cat(flat_tensors)
            prefix = f"global_{comp_type}"
            
            comp_stats = compute_activation_stats(all_comp_weights)
            for stat_name, value in comp_stats.items():
                metrics[f"{prefix}/{stat_name}"] = value
        
        # Compute global stats across all hidden layers
        flat_all_tensors = [w.reshape(-1) for w in all_layer_weights]
        all_weights = torch.cat(flat_all_tensors)
        
        # Add full global metrics
        global_stats = compute_activation_stats(all_weights)
        for stat_name, value in global_stats.items():
            metrics[f"global_global/{stat_name}"] = value
        
        return metrics
    
    def collect_parameter_metrics(self, model: torch.nn.Module, debug: bool = False) -> Dict[str, torch.Tensor]:
        """Collect metrics for model parameters."""
        metrics = {}
        max_layers = self.config.model.model_config.num_hidden_layers

        # Pre-format patterns for efficiency
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
                        stats = compute_activation_stats(param)
                        for stat_name, value in stats.items():
                            metrics[f"{layer_name}/{comp_type}/{subcomp_name}/{stat_name}"] = value
                        
                        # Add gradient stats if available
                        if hasattr(param, "grad") and param.grad is not None:
                            grad_stats = compute_activation_stats(param.grad.detach())
                            for stat_name, value in grad_stats.items():
                                metrics[f"{layer_name}/{comp_type}/{subcomp_name}/grad/{stat_name}"] = value

        # Get final layer norm
        final_ln = get_attribute_by_path(model, "model.final_layer_norm.pp_block.weight")
        if final_ln is not None:
            stats = compute_activation_stats(final_ln)
            for stat_name, value in stats.items():
                metrics[f"final_layernorm/{stat_name}"] = value

        return metrics
    
    def collect_accuracy_metrics(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Calculate token prediction accuracy metrics."""
        metrics = {}

        if logits is None or targets is None:
            return metrics

        # Ensure we're working with detached tensors
        logits = logits.detach() if hasattr(logits, "detach") else logits
        targets = targets.detach() if hasattr(targets, "detach") else targets

        # Check shapes
        if logits.dim() < 2 or targets.dim() < 1:
            return metrics

        # Get predictions
        predictions = logits.argmax(dim=-1)

        # Match shapes if needed
        if predictions.shape != targets.shape:
            if predictions.dim() > targets.dim():
                predictions = predictions[..., -targets.shape[-1]:]
            elif targets.dim() > predictions.dim():
                targets = targets[..., -predictions.shape[-1]:]

        # Calculate overall accuracy
        correct = (predictions == targets).float()
        accuracy = correct.mean()
        metrics["accuracy/overall"] = accuracy

        # Calculate next token accuracy
        if targets.dim() > 1 and targets.size(1) > 1:
            next_token_accuracy = correct[:, -1].mean()
            metrics["accuracy/next_token"] = next_token_accuracy

            # Position-specific accuracy
            for pos in range(min(5, targets.size(1))):
                pos_accuracy = correct[:, pos].mean()
                metrics[f"accuracy/position_{pos}"] = pos_accuracy

        return metrics
    
    def collect_all_metrics(
        self,
        model: torch.nn.Module,
        logits: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        iteration: Optional[int] = None,
        debug: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Collect all metrics based on the specified log level and iteration.
        """
        metrics = {}

        # Collect accuracy metrics if logits and targets are available
        if logits is not None and targets is not None:
            metrics.update(self.collect_accuracy_metrics(logits, targets))

        # Check if we should log detailed metrics
        if iteration is not None and self.should_log_detailed_metrics(iteration):
            try_collect = lambda fn, err_msg: (
                metrics.update(fn(model, debug=debug)) 
                if not debug else 
                try_with_debug(fn, model, debug, err_msg, metrics)
            )
            
            # Add global hidden layer metrics
            try_collect(
                self.compute_global_hidden_layer_metrics,
                "Error collecting global hidden layer metrics"
            )
                
            # Add embedding metrics
            try_collect(
                self.collect_embeddings_metrics,
                "Error collecting embedding metrics"
            )

            # If in full details mode (level=1)
            if self.log_level >= 1:
                # Add parameter metrics
                try_collect(
                    self.collect_parameter_metrics,
                    "Error collecting parameter metrics"
                )

        return metrics


def try_with_debug(fn, model, debug, err_msg, metrics):
    """Helper function to try collecting metrics with debug support"""
    try:
        metrics.update(fn(model, debug=debug))
    except Exception as e:
        print(f"{err_msg}: {e}")
    return metrics


if __name__ == "__main__":
    # Simple test code
    print("Metrics logging utilities loaded")
