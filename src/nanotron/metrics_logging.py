"""
Utility functions for computing and logging various model metrics.
"""

import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

########################
# Basic utility functions
########################


def compute_tensor_norm(tensor: torch.Tensor, p: int = 2) -> torch.Tensor:
    """Compute the Lp norm of a tensor."""
    flattened = tensor.reshape(-1)
    return torch.norm(flattened, p=p)


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


def compute_entropy(tensor: torch.Tensor) -> torch.Tensor:
    """Compute Shannon entropy of a tensor."""
    if tensor.min() < 0 or tensor.sum() != 1.0:
        if tensor.min() < 0:
            probs = F.softmax(tensor.float(), dim=-1)
        else:
            probs = tensor.float() / tensor.sum()
    else:
        probs = tensor.float()
    # Compute entropy: -sum(p * log(p))
    # Add epsilon to avoid log(0)
    entropy = -torch.sum(probs * torch.log2(probs + 1e-10))
    return entropy


def compute_kurtosis(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute kurtosis which measures "tailedness" of distribution.
    High kurtosis indicates outliers, low kurtosis indicates uniform-like distribution.
    """
    tensor = tensor.flatten().float()
    mean = tensor.mean()
    std = tensor.std()
    # Avoid division by zero
    if std == 0:
        return torch.tensor(0.0, device=tensor.device)

    # Compute kurtosis: E[((x-μ)/σ)^4]
    normalized = (tensor - mean) / std
    return torch.mean(normalized**4) - 3  # Excess kurtosis (normal distribution has kurtosis=3)


def compute_zero_fraction(tensor: torch.Tensor, threshold: float = 1e-10) -> torch.Tensor:
    """Compute fraction of near-zero elements."""
    return (tensor.abs() < threshold).float().mean()


def compute_activation_stats(tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Compute comprehensive statistics for a tensor."""
    if tensor.numel() == 0:
        return {
            "mean": torch.tensor(0.0, device=tensor.device),
            "std": torch.tensor(0.0, device=tensor.device),
            "min": torch.tensor(0.0, device=tensor.device),
            "max": torch.tensor(0.0, device=tensor.device),
            "norm_l1": torch.tensor(0.0, device=tensor.device),
            "norm_l2": torch.tensor(0.0, device=tensor.device),
            "zero_frac": torch.tensor(1.0, device=tensor.device),
            "kurtosis": torch.tensor(0.0, device=tensor.device),
        }

    tensor = tensor.float()
    mean = tensor.mean()
    var = tensor.var()
    std = torch.sqrt(var)
    min_val = tensor.min()
    max_val = tensor.max()
    norm_l1 = compute_tensor_norm(tensor, p=1)
    norm_l2 = compute_tensor_norm(tensor, p=2)
    zero_frac = compute_zero_fraction(tensor)
    kurtosis = compute_kurtosis(tensor)

    return {
        "mean": mean,
        "std": std,
        "min": min_val,
        "max": max_val,
        "norm_l1": norm_l1,
        "norm_l2": norm_l2,
        "zero_frac": zero_frac,
        "kurtosis": kurtosis,
    }


class ExperimentLogger:
    """
    Class for logging experiment metrics with configurable detail levels.
    
    Supports two modes:
    - Basic (level=0): Only logs essential metrics (loss, accuracy, performance)
    - Full (level=1): Logs all metrics including detailed model statistics
    """
    
    def __init__(self, config=None):
        """
        Initialize the logger with configuration.
        
        Args:
            config: Optional configuration object with metrics_logging attribute
        """
        self.log_level = 0  # Default: basic logging
        self.log_interval = 50  # Default: log every 50 steps
        
        if config is not None and hasattr(config, 'metrics_logging') and config.metrics_logging is not None:
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
    
    def collect_embeddings_metrics(self, model: torch.nn.Module, debug: bool = False) -> Dict[str, torch.Tensor]:
        """Collect metrics for model embeddings."""
        metrics = {}

        embedding_paths = [
            "model.token_position_embeddings.pp_block.token_embedding.weight",
            "model.lm_head.pp_block.weight",
        ]

        for path in embedding_paths:
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
        all_layer_weights = []
        
        # Find number of layers
        max_layers = 0
        for i in range(100):  # Safety limit
            if get_attribute_by_path(model, f"model.decoder.{i}.pp_block") is not None:
                max_layers = i + 1
            else:
                break
        
        if max_layers == 0:
            return metrics
        
        # Components to collect
        components = {
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
        
        # Group weights by component type
        component_weights = {comp_type: [] for comp_type in components}
        
        # Collect all weights from hidden layers
        for layer_idx in range(max_layers):
            for component_type, subcomponents in components.items():
                for subcomp_name, pattern in subcomponents.items():
                    param = get_attribute_by_path(model, pattern.format(layer_idx))
                    if param is not None:
                        param_tensor = param.detach().float()
                        all_layer_weights.append(param_tensor)
                        component_weights[component_type].append(param_tensor)
        
        if not all_layer_weights:
            return metrics
        
        # Compute statistics for each component type
        for component_type, weights in component_weights.items():
            if weights:
                all_comp_weights = torch.cat([w.flatten() for w in weights])
                prefix = f"global_{component_type}"
                
                metrics[f"{prefix}/mean"] = all_comp_weights.mean()
                metrics[f"{prefix}/std"] = all_comp_weights.std()
                metrics[f"{prefix}/min"] = all_comp_weights.min()
                metrics[f"{prefix}/max"] = all_comp_weights.max()
                metrics[f"{prefix}/norm_l1"] = compute_tensor_norm(all_comp_weights, p=1)
                metrics[f"{prefix}/norm_l2"] = compute_tensor_norm(all_comp_weights, p=2)
                metrics[f"{prefix}/zero_frac"] = compute_zero_fraction(all_comp_weights)
                metrics[f"{prefix}/kurtosis"] = compute_kurtosis(all_comp_weights)
        
        # Compute global stats across all hidden layers
        all_weights = torch.cat([w.flatten() for w in all_layer_weights])
        
        # Add full global metrics
        metrics["global_global/mean"] = all_weights.mean()
        metrics["global_global/std"] = all_weights.std()
        metrics["global_global/min"] = all_weights.min()
        metrics["global_global/max"] = all_weights.max()
        metrics["global_global/norm_l1"] = compute_tensor_norm(all_weights, p=1)
        metrics["global_global/norm_l2"] = compute_tensor_norm(all_weights, p=2)
        metrics["global_global/zero_frac"] = compute_zero_fraction(all_weights)
        metrics["global_global/kurtosis"] = compute_kurtosis(all_weights)
        
        return metrics
    
    def collect_attention_metrics(self, model: torch.nn.Module, debug: bool = False) -> Dict[str, torch.Tensor]:
        """Collect metrics for attention weights and patterns."""
        metrics = {}

        # Find attention weights and patterns in model
        weights_pattern = "model.decoder.{}.pp_block.attn.attention.weights"
        scores_pattern = "model.decoder.{}.pp_block.attn.attention.scores"
        qkv_pattern = "model.decoder.{}.pp_block.attn.qkv_proj.weight"
        o_pattern = "model.decoder.{}.pp_block.attn.o_proj.weight"

        # Find number of layers
        max_layers = 0
        for i in range(100):  # Safety limit
            if any(get_attribute_by_path(model, pattern.format(i)) is not None 
                    for pattern in [weights_pattern, scores_pattern, qkv_pattern, o_pattern]):
                max_layers = i + 1
            else:
                break

        # Check weights at each layer
        for layer_idx in range(max_layers):
            layer_name = f"layer_{layer_idx}"
            
            for name, pattern in [
                ("weights", weights_pattern), 
                ("scores", scores_pattern),
                ("qkv_weights", qkv_pattern),
                ("o_weights", o_pattern)
            ]:
                tensor = get_attribute_by_path(model, pattern.format(layer_idx))
                if tensor is not None:
                    stats = compute_activation_stats(tensor)
                    for stat_name, value in stats.items():
                        metrics[f"{layer_name}/attention/{name}/{stat_name}"] = value
                    
                    # Calculate entropy for attention distributions
                    if name == "weights" and tensor.dim() >= 2:
                        entropy = compute_entropy(tensor)
                        metrics[f"{layer_name}/attention/{name}/entropy"] = entropy

        return metrics
    
    def collect_parameter_metrics(self, model: torch.nn.Module, debug: bool = False) -> Dict[str, torch.Tensor]:
        """Collect metrics for model parameters."""
        metrics = {}

        # Define layer component patterns
        components = {
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

        # Find number of layers
        max_layers = 0
        for i in range(100):  # Safety limit
            if any(get_attribute_by_path(model, pattern.format(i)) is not None 
                for comp in components.values() for pattern in comp.values()):
                max_layers = i + 1
            else:
                break

        # Collect metrics for each layer
        for layer_idx in range(max_layers):
            layer_name = f"layer_{layer_idx}"
            
            for component_type, subcomponents in components.items():
                for subcomp_name, pattern in subcomponents.items():
                    param = get_attribute_by_path(model, pattern.format(layer_idx))
                    
                    if param is not None:
                        stats = compute_activation_stats(param)
                        
                        for stat_name, value in stats.items():
                            metrics[f"{layer_name}/{component_type}/{subcomp_name}/{stat_name}"] = value
                        
                        # Add gradient stats if available
                        if hasattr(param, "grad") and param.grad is not None:
                            grad = param.grad.detach()
                            grad_stats = compute_activation_stats(grad)
                            
                            for stat_name, value in grad_stats.items():
                                metrics[f"{layer_name}/{component_type}/{subcomp_name}/grad/{stat_name}"] = value

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
    
    def collect_loss_metrics(self, loss: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Collect metrics related to the loss."""
        metrics = {}

        if loss is None:
            return metrics

        # Ensure detached tensor
        loss = loss.detach() if hasattr(loss, "detach") else loss
        metrics["loss/value"] = loss

        return metrics
    
    def collect_performance_metrics(self, step_time: float, batch_size: int, seq_length: int) -> Dict[str, torch.Tensor]:
        """Collect performance metrics like throughput."""
        metrics = {}
        device = "cpu"  # Performance metrics can be on CPU

        metrics["performance/step_time"] = torch.tensor(step_time, device=device)
        
        tokens_per_batch = batch_size * seq_length
        tokens_per_second = tokens_per_batch / step_time if step_time > 0 else 0
        metrics["performance/tokens_per_second"] = torch.tensor(tokens_per_second, device=device)
        
        return metrics
    
    def collect_all_metrics(
        self,
        model: torch.nn.Module,
        logits: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        loss: Optional[torch.Tensor] = None,
        step_time: Optional[float] = None,
        batch_size: Optional[int] = None,
        seq_length: Optional[int] = None,
        iteration: Optional[int] = None,
        debug: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Collect all metrics based on the specified log level and iteration.
        
        Args:
            model: The model to collect metrics from
            logits: Model output logits
            targets: Target tokens
            loss: Loss value
            step_time: Training step time in seconds
            batch_size: Batch size
            seq_length: Sequence length
            iteration: Current training iteration/step
            debug: Whether to print debug information
            
        Returns:
            Dict of metrics
        """
        metrics = {}

        # Always collect basic metrics
        if loss is not None:
            metrics.update(self.collect_loss_metrics(loss))

        if step_time is not None and batch_size is not None and seq_length is not None:
            metrics.update(self.collect_performance_metrics(step_time, batch_size, seq_length))

        # Collect accuracy metrics if logits and targets are available
        if logits is not None and targets is not None:
            metrics.update(self.collect_accuracy_metrics(logits, targets))

        # Check if we should log detailed metrics
        if iteration is not None and self.should_log_detailed_metrics(iteration):
            # Add global hidden layer metrics
            try:
                metrics.update(self.compute_global_hidden_layer_metrics(model, debug=debug))
            except Exception as e:
                if debug:
                    print(f"Error collecting global hidden layer metrics: {e}")
                    
            # Add embedding metrics
            try:
                metrics.update(self.collect_embeddings_metrics(model, debug=debug))
            except Exception as e:
                if debug:
                    print(f"Error collecting embedding metrics: {e}")

            # If in full details mode (level=1)
            if self.log_level >= 1:
                # Add attention metrics
                try:
                    metrics.update(self.collect_attention_metrics(model, debug=debug))
                except Exception as e:
                    if debug:
                        print(f"Error collecting attention metrics: {e}")

                # Add parameter metrics
                try:
                    metrics.update(self.collect_parameter_metrics(model, debug=debug))
                except Exception as e:
                    if debug:
                        print(f"Error collecting parameter metrics: {e}")

        return metrics


if __name__ == "__main__":
    # Simple test code
    print("Metrics logging utilities loaded")
