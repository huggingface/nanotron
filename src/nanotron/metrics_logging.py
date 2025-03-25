"""
Utility functions for computing and logging various model metrics.
Organized by metric category with consistent path handling.
"""

import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

########################
# Basic utility functions
########################


def compute_tensor_norm(tensor: torch.Tensor, p: int = 2) -> torch.Tensor:
    """
    Compute the Lp norm of a tensor.

    Args:
        tensor (torch.Tensor): The input tensor
        p (int, optional): The norm degree. Default: 2 for L2 norm

    Returns:
        torch.Tensor: A scalar tensor containing the norm
    """
    # Flatten the tensor to compute norm over all elements
    flattened = tensor.reshape(-1)

    # Compute and return the norm
    return torch.norm(flattened, p=p)


def get_attribute_by_path(obj: Any, path: str, debug: bool = False) -> Optional[Any]:
    """
    Get an attribute from an object by a dot-separated path.

    Args:
        obj: The object to get attributes from
        path: A dot-separated path to the attribute
        debug: Whether to print debug info when attribute is not found

    Returns:
        The attribute if found, None otherwise
    """
    # Split the path into attribute components
    path_parts = path.split(".")

    # Navigate through the object attributes
    current = obj
    for i, attr in enumerate(path_parts):
        if hasattr(current, attr):
            current = getattr(current, attr)
        else:
            if debug:
                print(f"Could not find attribute '{attr}' in path '{path}' at position {i}")
            return None

    # Always detach tensor to prevent computing graph issues
    if isinstance(current, torch.Tensor) and hasattr(current, "detach"):
        return current.detach()

    return current


########################
# Statistical functions
########################


def compute_entropy(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute Shannon entropy of a tensor.

    Args:
        tensor: Input tensor

    Returns:
        Entropy value
    """
    # Normalize to probabilities if needed
    if tensor.min() < 0 or tensor.sum() != 1.0:
        # For logits, use softmax
        if tensor.min() < 0:
            probs = F.softmax(tensor.float(), dim=-1)
        else:
            # Otherwise normalize directly
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

    Args:
        tensor: Input tensor

    Returns:
        Kurtosis value
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
    """
    Compute fraction of near-zero elements.

    Args:
        tensor: Input tensor
        threshold: Threshold below which values are considered zero

    Returns:
        Fraction of zero elements (0.0 to 1.0)
    """
    return (tensor.abs() < threshold).float().mean()


def compute_activation_stats(tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Compute comprehensive statistics for an activation or parameter tensor.

    Args:
        tensor: Input tensor

    Returns:
        Dictionary of statistics
    """
    # Handle empty tensor
    if tensor.numel() == 0:
        zero_stats = {
            "mean": torch.tensor(0.0, device=tensor.device),
            "std": torch.tensor(0.0, device=tensor.device),
            "var": torch.tensor(0.0, device=tensor.device),
            "min": torch.tensor(0.0, device=tensor.device),
            "max": torch.tensor(0.0, device=tensor.device),
            "norm_l1": torch.tensor(0.0, device=tensor.device),
            "norm_l2": torch.tensor(0.0, device=tensor.device),
            "abs_mean": torch.tensor(0.0, device=tensor.device),
            "zero_frac": torch.tensor(1.0, device=tensor.device),
            "kurtosis": torch.tensor(0.0, device=tensor.device),
        }
        return zero_stats

    # Basic statistics
    tensor = tensor.float()  # Ensure float for statistics
    mean = tensor.mean()
    var = tensor.var()
    std = torch.sqrt(var)
    min_val = tensor.min()
    max_val = tensor.max()

    # Norms
    flattened = tensor.reshape(-1)
    norm_l1 = torch.norm(flattened, p=1)
    norm_l2 = torch.norm(flattened, p=2)

    # Other metrics
    abs_mean = tensor.abs().mean()
    zero_frac = compute_zero_fraction(tensor)

    # Advanced statistics
    kurtosis = compute_kurtosis(tensor)

    return {
        "mean": mean,
        "std": std,
        "var": var,
        "min": min_val,
        "max": max_val,
        "norm_l1": norm_l1,
        "norm_l2": norm_l2,
        "abs_mean": abs_mean,
        "zero_frac": zero_frac,
        "kurtosis": kurtosis,
    }


def compute_covariance_off_diag_sum(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute sum of off-diagonal elements of covariance matrix.
    This is a signal propagation metric - measures correlation between activations.

    Args:
        tensor: Input tensor with shape [batch_size, features]

    Returns:
        Sum of off-diagonal elements
    """
    # Ensure tensor has at least 2 dimensions
    if tensor.dim() < 2:
        return torch.tensor(0.0, device=tensor.device)

    # For higher dim tensors, flatten all but the last dimension
    if tensor.dim() > 2:
        tensor = tensor.reshape(-1, tensor.size(-1))

    # Features must be > 1 to have off-diagonal elements
    if tensor.size(1) <= 1:
        return torch.tensor(0.0, device=tensor.device)

    # Compute covariance matrix
    tensor = tensor - tensor.mean(dim=0, keepdim=True)
    cov = torch.matmul(tensor.t(), tensor) / (tensor.size(0) - 1)

    # Sum off-diagonal elements
    off_diag_sum = cov.sum() - cov.diag().sum()

    return off_diag_sum


def compute_gradient_stats(param: torch.nn.Parameter) -> Dict[str, torch.Tensor]:
    """
    Compute statistics about a parameter's gradient.

    Args:
        param: Parameter to analyze

    Returns:
        Dict of gradient statistics
    """
    if param.grad is None:
        return {
            "grad_mean": torch.tensor(0.0, device=param.device),
            "grad_std": torch.tensor(0.0, device=param.device),
            "grad_norm_l1": torch.tensor(0.0, device=param.device),
            "grad_norm_l2": torch.tensor(0.0, device=param.device),
            "grad_zero_frac": torch.tensor(1.0, device=param.device),
        }

    grad = param.grad.detach()

    # Basic statistics
    grad_mean = grad.mean()
    grad_var = grad.var()
    grad_std = torch.sqrt(grad_var)

    # Norms
    grad_norm_l1 = torch.norm(grad.reshape(-1), p=1)
    grad_norm_l2 = torch.norm(grad.reshape(-1), p=2)

    # Zero fraction
    grad_zero_frac = compute_zero_fraction(grad)

    return {
        "grad_mean": grad_mean,
        "grad_std": grad_std,
        "grad_norm_l1": grad_norm_l1,
        "grad_norm_l2": grad_norm_l2,
        "grad_zero_frac": grad_zero_frac,
    }


def compute_embedding_norm(
    model: Any, embedding_path: str = "token_position_embeddings.pp_block.token_embedding.weight", p: int = 2
) -> torch.Tensor:
    """
    Compute the Lp norm of a model's embedding weights.

    Args:
        model: The model containing embeddings
        embedding_path (str): Attribute path to the embedding weights
        p (int, optional): The norm degree. Default: 2 for L2 norm

    Returns:
        torch.Tensor: A scalar tensor containing the norm
    """
    embedding = get_attribute_by_path(model, embedding_path)
    if embedding is None:
        raise AttributeError(f"Could not find embedding at path {embedding_path}")

    return compute_tensor_norm(embedding, p=p)


def compute_param_norm_stats(
    model: torch.nn.Module, param_filter: Optional[callable] = None
) -> Dict[str, torch.Tensor]:
    """
    Compute statistics about parameter norms in a model.

    Args:
        model (torch.nn.Module): The model to analyze
        param_filter (callable, optional): Function that takes param name and param,
                                           returns True to include the parameter

    Returns:
        Dict containing various norm statistics:
            - mean_norm: Average L2 norm across parameters
            - max_norm: Maximum L2 norm
            - min_norm: Minimum L2 norm
            - total_norm: L2 norm of all parameters combined
    """
    norms = []
    total_squared = 0.0

    for name, param in model.named_parameters():
        if param_filter is not None and not param_filter(name, param):
            continue

        if param.requires_grad:
            norm = compute_tensor_norm(param).item()
            norms.append(norm)
            total_squared += norm**2

    if not norms:
        return {
            "mean_norm": torch.tensor(0.0),
            "max_norm": torch.tensor(0.0),
            "min_norm": torch.tensor(0.0),
            "total_norm": torch.tensor(0.0),
        }

    return {
        "mean_norm": torch.tensor(sum(norms) / len(norms)),
        "max_norm": torch.tensor(max(norms)),
        "min_norm": torch.tensor(min(norms)),
        "total_norm": torch.tensor(total_squared**0.5),  # sqrt of sum of squared norms
    }


def collect_embedding_metrics(model: Any, embedding_paths: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
    """
    Collect metrics for multiple embeddings in a model.

    Args:
        model: The model containing embeddings
        embedding_paths: List of attribute paths to embeddings to analyze.
                         If None, uses a default set of common embedding paths.

    Returns:
        Dict mapping metric names to values
    """
    if embedding_paths is None:
        # Default paths for LLaMA model structure, including complex paths
        embedding_paths = [
            # Main token embeddings - common paths
            "token_position_embeddings.pp_block.token_embedding.weight",
            "model.token_position_embeddings.pp_block.token_embedding.weight",
            "model.model.token_position_embeddings.pp_block.token_embedding.weight",
            # With additional operations
            "model.model.token_position_embeddings.pp_block.token_embedding.weight.detach",
            # LM head weights
            "lm_head.pp_block.weight",
            # Other common paths
            "model.embed_tokens.weight",
            "model.embeddings.word_embeddings.weight",
        ]

    metrics = {}

    for path in embedding_paths:
        try:
            embedding = get_attribute_by_path(model, path)

            if embedding is not None and isinstance(embedding, torch.Tensor):
                # Get the name for the metric (last two parts of the path without operations)
                clean_path = path.split(".detach")[0].split(".H")[0]  # Remove operations
                parts = clean_path.split(".")
                name = "_".join(parts[-2:]) if len(parts) > 1 else path

                # Add norm metric
                metrics[f"{name}_norm"] = compute_tensor_norm(embedding)

                # Calculate other embedding statistics
                stats = compute_activation_stats(embedding)
                for stat_name, value in stats.items():
                    metrics[f"{name}_{stat_name}"] = value

        except (AttributeError, ValueError):
            # Skip if the embedding path doesn't exist
            pass

    return metrics


def collect_attention_metrics(
    model: Any, attention_paths: Optional[List[str]] = None, sample_layers: Optional[List[int]] = None
) -> Dict[str, torch.Tensor]:
    """
    Collect metrics for attention matrices in a model.

    Args:
        model: The model containing attention modules
        attention_paths: List of attribute paths to attention modules to analyze.
                         If None, uses default paths based on model structure.
        sample_layers: List of layer indices to sample from. If None, samples from
                       first, middle, and last layer.

    Returns:
        Dict mapping metric names to values
    """
    metrics = {}

    # If attention_paths is None, try to detect them based on model architecture
    if attention_paths is None:
        # For LLaMA models
        attention_paths = []

        # Try to detect how many layers the model has
        if hasattr(model, "decoder") and hasattr(model.decoder, "_modules"):
            num_layers = len(model.decoder._modules)

            # If sample_layers is None, sample first, middle, and last layer if available
            if sample_layers is None:
                sample_layers = [0]
                if num_layers > 2:
                    sample_layers.append(num_layers // 2)
                if num_layers > 1:
                    sample_layers.append(num_layers - 1)

            # Generate attention paths for the selected layers
            for layer_idx in sample_layers:
                if layer_idx < num_layers:
                    # LLaMA model structure - with different possible prefixes
                    prefixes = ["", "model.", "model.model."]
                    for prefix in prefixes:
                        attention_paths.append(f"{prefix}decoder.{layer_idx}.pp_block.attn.qkv_proj.weight")
                        attention_paths.append(f"{prefix}decoder.{layer_idx}.pp_block.attn.o_proj.weight")

    for path in attention_paths:
        try:
            weights = get_attribute_by_path(model, path)

            if weights is not None and isinstance(weights, torch.Tensor):
                # Generate a more readable name
                clean_path = path.split(".detach")[0].split(".H")[0]  # Remove operations
                parts = clean_path.split(".")

                # Get the layer index and component name
                if "decoder" in parts:
                    layer_idx = parts[parts.index("decoder") + 1]
                    attn_part = parts[-2]  # qkv_proj or o_proj
                    name = f"layer{layer_idx}_{attn_part}"
                else:
                    name = "_".join(parts[-2:])

                # Calculate statistics for the attention weights
                stats = compute_activation_stats(weights)
                for stat_name, value in stats.items():
                    metrics[f"{name}_{stat_name}"] = value

        except (AttributeError, ValueError, IndexError):
            # Skip if the weights don't exist
            pass

    return metrics


def collect_projection_metrics(
    model: Any, projection_paths: Optional[List[str]] = None, sample_layers: Optional[List[int]] = None
) -> Dict[str, torch.Tensor]:
    """
    Collect metrics for projection matrices in the model's MLP blocks.

    Args:
        model: The model containing MLP projections
        projection_paths: List of attribute paths to MLP projections.
                          If None, uses default paths based on model structure.
        sample_layers: List of layer indices to sample from. If None, samples from
                       first, middle, and last layer.

    Returns:
        Dict mapping metric names to values
    """
    metrics = {}

    # If projection_paths is None, try to detect them based on model architecture
    if projection_paths is None:
        # For LLaMA models
        projection_paths = []

        # Try to detect how many layers the model has
        if hasattr(model, "decoder") and hasattr(model.decoder, "_modules"):
            num_layers = len(model.decoder._modules)

            # If sample_layers is None, sample first, middle, and last layer if available
            if sample_layers is None:
                sample_layers = [0]
                if num_layers > 2:
                    sample_layers.append(num_layers // 2)
                if num_layers > 1:
                    sample_layers.append(num_layers - 1)

            # Generate projection paths for the selected layers with different possible prefixes
            for layer_idx in sample_layers:
                if layer_idx < num_layers:
                    # LLaMA model structure with different prefixes
                    prefixes = ["", "model.", "model.model."]
                    for prefix in prefixes:
                        projection_paths.append(f"{prefix}decoder.{layer_idx}.pp_block.mlp.gate_up_proj.weight")
                        projection_paths.append(f"{prefix}decoder.{layer_idx}.pp_block.mlp.down_proj.weight")

    for path in projection_paths:
        try:
            weights = get_attribute_by_path(model, path)

            if weights is not None and isinstance(weights, torch.Tensor):
                # Generate a more readable name
                clean_path = path.split(".detach")[0].split(".H")[0]  # Remove operations
                parts = clean_path.split(".")

                # Get the layer index and component name
                if "decoder" in parts:
                    layer_idx = parts[parts.index("decoder") + 1]
                    mlp_part = parts[-2]  # gate_up_proj or down_proj
                    name = f"layer{layer_idx}_{mlp_part}"
                else:
                    name = "_".join(parts[-2:])

                # Calculate statistics for the projection weights
                stats = compute_activation_stats(weights)
                for stat_name, value in stats.items():
                    metrics[f"{name}_{stat_name}"] = value

        except (AttributeError, ValueError, IndexError):
            # Skip if the weights don't exist
            pass

    return metrics


def get_layernorm_metrics(
    model: Any, layernorm_paths: Optional[List[str]] = None, sample_layers: Optional[List[int]] = None
) -> Dict[str, torch.Tensor]:
    """
    Collect metrics for LayerNorm/RMSNorm parameters.

    Args:
        model: The model containing layer norms
        layernorm_paths: List of attribute paths to layer norms.
                        If None, uses default paths based on model structure.
        sample_layers: List of layer indices to sample from. If None, samples from
                       first, middle, and last layer.

    Returns:
        Dict mapping metric names to values
    """
    metrics = {}

    # If layernorm_paths is None, try to detect them based on model architecture
    if layernorm_paths is None:
        # For LLaMA models
        layernorm_paths = []

        # Try to detect how many layers the model has
        if hasattr(model, "decoder") and hasattr(model.decoder, "_modules"):
            num_layers = len(model.decoder._modules)

            # If sample_layers is None, sample first, middle, and last layer if available
            if sample_layers is None:
                sample_layers = [0]
                if num_layers > 2:
                    sample_layers.append(num_layers // 2)
                if num_layers > 1:
                    sample_layers.append(num_layers - 1)

            # Generate layernorm paths for the selected layers with different prefixes
            prefixes = ["", "model.", "model.model."]
            for layer_idx in sample_layers:
                if layer_idx < num_layers:
                    # LLaMA model structure - RMSNorm
                    for prefix in prefixes:
                        layernorm_paths.append(f"{prefix}decoder.{layer_idx}.pp_block.input_layernorm.weight")
                        layernorm_paths.append(f"{prefix}decoder.{layer_idx}.pp_block.post_attention_layernorm.weight")

            # Also add the final layernorm with different prefixes
            for prefix in prefixes:
                layernorm_paths.append(f"{prefix}final_layer_norm.pp_block.weight")

    for path in layernorm_paths:
        try:
            weights = get_attribute_by_path(model, path)

            if weights is not None and isinstance(weights, torch.Tensor):
                # Generate a more readable name
                clean_path = path.split(".detach")[0].split(".H")[0]  # Remove operations
                parts = clean_path.split(".")

                # Get component name
                if "decoder" in parts and "pp_block" in parts:
                    layer_idx = parts[parts.index("decoder") + 1]
                    ln_idx = parts.index("pp_block") + 1
                    if ln_idx < len(parts):
                        ln_type = parts[ln_idx]  # input_layernorm or post_attention_layernorm
                        name = f"layer{layer_idx}_{ln_type}"
                    else:
                        name = f"layer{layer_idx}_norm"
                elif "final_layer_norm" in parts:
                    name = "final_layernorm"
                else:
                    name = "_".join(parts[-2:])

                # Calculate statistics for the layernorm weights
                stats = compute_activation_stats(weights)
                for stat_name, value in stats.items():
                    metrics[f"{name}_{stat_name}"] = value

        except (AttributeError, ValueError, IndexError):
            # Skip if the weights don't exist
            pass

    return metrics


def log_model_metrics(
    model: torch.nn.Module,
    logger: callable,
    include_embeddings: bool = True,
    include_param_stats: bool = True,
    include_attention: bool = True,
    include_projections: bool = True,
    include_layernorms: bool = True,
    log_level: str = "full",  # Options: "minimal", "medium", "full"
) -> Dict[str, torch.Tensor]:
    """
    Compute and log comprehensive metrics about a model.

    Args:
        model: The model to analyze
        logger: Callable that will be called with metric name and value
        include_embeddings: Whether to include embedding metrics
        include_param_stats: Whether to include parameter statistics
        include_attention: Whether to include attention metrics
        include_projections: Whether to include projection metrics
        include_layernorms: Whether to include layernorm metrics
        log_level: Level of detail in logging - "minimal", "medium", or "full"

    Returns:
        Dict of all metrics computed
    """
    metrics = {}

    # Unwrap model if needed
    if hasattr(model, "module"):
        unwrapped_model = model.module
    else:
        unwrapped_model = model

    # Set layer sampling based on log_level
    if log_level == "minimal":
        sample_layers = [0]  # Just the first layer
    elif log_level == "medium":
        # Get number of layers
        num_layers = len(unwrapped_model.decoder._modules) if hasattr(unwrapped_model, "decoder") else 0
        # First and last layer
        sample_layers = [0, num_layers - 1] if num_layers > 1 else [0]
    else:  # "full"
        sample_layers = None  # Will use default sampling in the individual functions

    # Get embedding metrics if requested
    if include_embeddings:
        embedding_metrics = collect_embedding_metrics(unwrapped_model)
        metrics.update(embedding_metrics)

    # Get attention metrics if requested
    if include_attention:
        attention_metrics = collect_attention_metrics(unwrapped_model, sample_layers=sample_layers)
        metrics.update(attention_metrics)

    # Get projection metrics if requested
    if include_projections:
        projection_metrics = collect_projection_metrics(unwrapped_model, sample_layers=sample_layers)
        metrics.update(projection_metrics)

    # Get layernorm metrics if requested
    if include_layernorms:
        layernorm_metrics = get_layernorm_metrics(unwrapped_model, sample_layers=sample_layers)
        metrics.update(layernorm_metrics)

    # Get general parameter statistics if requested
    if include_param_stats:
        # All parameters
        param_metrics = compute_param_norm_stats(unwrapped_model)
        for name, value in param_metrics.items():
            metrics[f"all_params_{name}"] = value

        # Just weights (excluding biases)
        weight_metrics = compute_param_norm_stats(
            unwrapped_model, param_filter=lambda name, _: "weight" in name.split(".")[-1]
        )
        for name, value in weight_metrics.items():
            metrics[f"weights_{name}"] = value

        # Just biases
        bias_metrics = compute_param_norm_stats(
            unwrapped_model, param_filter=lambda name, _: "bias" in name.split(".")[-1]
        )
        for name, value in bias_metrics.items():
            metrics[f"biases_{name}"] = value

    # Log all collected metrics
    for name, value in metrics.items():
        if isinstance(value, torch.Tensor):
            value = value.item()
        logger(name, value)

    return metrics


########################
# Environment variables and logging configuration
########################

METRICS_LOG_INTERVAL = int(os.environ.get("METRICS_LOG_INTERVAL", "50"))


def get_metrics_log_level() -> int:
    """
    Get the metrics logging level from the FULL_LOGS environment variable.

    Returns:
        0: basic logging (default)
        1: medium detail
        2: full detail
    """
    return int(os.environ.get("FULL_LOGS", "0"))


def should_log_detailed_metrics(iteration: int) -> bool:
    """
    Determine if detailed metrics should be logged for the current iteration.

    Args:
        iteration: Current training iteration

    Returns:
        True if detailed metrics should be logged, False otherwise
    """
    # Always log first few iterations for debugging
    if iteration < 5:
        return True

    # Log according to interval if FULL_LOGS environment variable is set
    log_level = get_metrics_log_level()
    if log_level > 0 and iteration % METRICS_LOG_INTERVAL == 0:
        return True

    return False


########################
# Metrics collection functions
########################


def collect_embeddings_metrics(
    model: torch.nn.Module, prefix: str = "embeddings", debug: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Collect metrics for model embeddings (input and output).

    Args:
        model: The model to analyze
        prefix: Metrics name prefix
        debug: Whether to print debug info

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Use consistent path pattern for LLaMA model structure
    embedding_paths = [
        "model.token_position_embeddings.pp_block.token_embedding.weight",  # Input embeddings
        "model.lm_head.pp_block.weight",  # Output embeddings
    ]

    for path in embedding_paths:
        embeddings = get_attribute_by_path(model, path, debug=debug)
        if embeddings is not None:
            # Extract the name from the path for the metric name
            name = path.split(".")[-2]

            # Calculate comprehensive stats
            stats = compute_activation_stats(embeddings)

            # Add metrics with proper grouping
            for stat_name, value in stats.items():
                metric_name = f"{prefix}/{name}/{stat_name}"
                metrics[metric_name] = value

            # Add the tensor size as a separate metric
            metrics[f"{prefix}/{name}/size"] = torch.tensor(embeddings.numel(), device=embeddings.device)
        elif debug:
            print(f"Warning: Could not find embeddings at path '{path}'")

    return metrics


def collect_activation_metrics(
    model: torch.nn.Module, prefix: str = "activations", debug: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Collect metrics for model activations.

    Args:
        model: The model to analyze
        prefix: Metrics name prefix
        debug: Whether to print debug info

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Use consistent path pattern for LLaMA model activations
    base_decoder_paths = [
        "model.decoder.{}.pp_block.mlp.split_silu_mul.activation",
        "model.decoder.{}.pp_block.attn.attention.output",
    ]

    # Find the number of layers dynamically
    max_layers = 0
    for i in range(100):  # Safety limit
        found = False
        for pattern in base_decoder_paths:
            path = pattern.format(i)
            activation = get_attribute_by_path(model, path, debug=False)
            if activation is not None:
                found = True
                break

        if found:
            max_layers = i + 1
        else:
            break

    if debug:
        print(f"Found {max_layers} decoder layers")

    # Collect metrics for each layer and activation type
    for layer_idx in range(max_layers):
        for pattern in base_decoder_paths:
            path = pattern.format(layer_idx)
            activation = get_attribute_by_path(model, path, debug=False)

            if activation is None:
                continue

            # Calculate stats for this activation
            stats = compute_activation_stats(activation)

            # Add metrics with proper grouping by layer and component
            path_parts = path.split(".")
            layer_name = f"layer_{layer_idx}"

            # Determine the component type from the path
            if "mlp" in path:
                component = "mlp"
            elif "attn" in path:
                component = "attention"
            else:
                component = path_parts[-2] if len(path_parts) > 2 else "unknown"

            for stat_name, value in stats.items():
                metric_name = f"{prefix}/{layer_name}/{component}/{stat_name}"
                metrics[metric_name] = value

            # Add shape information
            metrics[f"{prefix}/{layer_name}/{component}/shape"] = torch.tensor(
                activation.numel(), device=activation.device
            )

    return metrics


def collect_attention_metrics(
    model: torch.nn.Module, prefix: str = "attention", debug: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Collect metrics for attention weights and patterns.

    Args:
        model: The model to analyze
        prefix: Metrics name prefix
        debug: Whether to print debug info

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Use consistent path pattern for LLaMA model attention
    weights_pattern = "model.decoder.{}.pp_block.attn.attention.weights"
    scores_pattern = "model.decoder.{}.pp_block.attn.attention.scores"

    # Find the number of layers dynamically
    max_layers = 0
    for i in range(100):  # Safety limit
        weights_path = weights_pattern.format(i)
        scores_path = scores_pattern.format(i)
        weights = get_attribute_by_path(model, weights_path, debug=False)
        scores = get_attribute_by_path(model, scores_path, debug=False)

        if weights is not None or scores is not None:
            max_layers = i + 1
        else:
            break

    if debug and max_layers == 0:
        print("Warning: Could not find any attention weights or scores.")

    # Check for attention scores and patterns at each layer
    for layer_idx in range(max_layers):
        layer_name = f"layer_{layer_idx}"

        # Attention weights path (post-softmax attention weights)
        attn_weights_path = weights_pattern.format(layer_idx)
        attn_weights = get_attribute_by_path(model, attn_weights_path, debug=False)

        # Attention scores path (pre-softmax attention scores/logits)
        attn_scores_path = scores_pattern.format(layer_idx)
        attn_scores = get_attribute_by_path(model, attn_scores_path, debug=False)

        # If both are None, we've reached the end of layers or data not available
        if attn_weights is None and attn_scores is None and layer_idx == 0 and debug:
            print(f"Warning: Could not find attention data at layer {layer_idx}")
            continue

        # Process attention weights if available
        if attn_weights is not None:
            # Calculate stats
            stats = compute_activation_stats(attn_weights)

            # Add metrics
            for stat_name, value in stats.items():
                metric_name = f"{prefix}/{layer_name}/weights/{stat_name}"
                metrics[metric_name] = value

            # Calculate entropy of attention distribution
            if attn_weights.dim() >= 2:
                attention_entropy = compute_entropy(attn_weights)
                metrics[f"{prefix}/{layer_name}/weights/entropy"] = attention_entropy

        # Process attention scores/logits if available
        if attn_scores is not None:
            # Calculate stats
            stats = compute_activation_stats(attn_scores)

            # Add metrics
            for stat_name, value in stats.items():
                metric_name = f"{prefix}/{layer_name}/scores/{stat_name}"
                metrics[metric_name] = value

    return metrics


def collect_parameter_metrics(
    model: torch.nn.Module, prefix: str = "parameters", debug: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Collect metrics for model parameters by layer and component.

    Args:
        model: The model to analyze
        prefix: Metrics name prefix
        debug: Whether to print debug info

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Use consistent path patterns for LLaMA model parameters
    component_patterns = {
        "attention": {
            "qkv_proj": "model.decoder.{}.pp_block.attn.qkv_proj.{}",
            "o_proj": "model.decoder.{}.pp_block.attn.o_proj.{}",
        },
        "mlp": {
            "gate_up_proj": "model.decoder.{}.pp_block.mlp.gate_up_proj.{}",
            "down_proj": "model.decoder.{}.pp_block.mlp.down_proj.{}",
        },
        "norm": {
            "input_layernorm": "model.decoder.{}.pp_block.input_layernorm.{}",
            "post_attention_layernorm": "model.decoder.{}.pp_block.post_attention_layernorm.{}",
        },
    }

    # Find the number of layers dynamically
    max_layers = 0
    for i in range(100):  # Safety limit
        found = False
        for component_type, subcomponents in component_patterns.items():
            for subcomp_name, pattern in subcomponents.items():
                for param_type in ["weight", "bias"]:
                    path = pattern.format(i, param_type)
                    param = get_attribute_by_path(model, path, debug=False)
                    if param is not None:
                        found = True
                        break
                if found:
                    break
            if found:
                break

        if found:
            max_layers = i + 1
        else:
            break

    if debug:
        print(f"Found {max_layers} parameter layers")

    # Collect metrics for each layer
    for layer_idx in range(max_layers):
        layer_name = f"layer_{layer_idx}"

        # Process each component type
        for component_type, subcomponents in component_patterns.items():
            for subcomp_name, pattern in subcomponents.items():
                # Look for weight and bias
                for param_type in ["weight", "bias"]:
                    path = pattern.format(layer_idx, param_type)
                    param = get_attribute_by_path(model, path, debug=False)

                    if param is not None:
                        # Calculate stats for parameter
                        param_stats = compute_activation_stats(param)

                        # Add metrics with proper organization
                        for stat_name, value in param_stats.items():
                            metric_name = (
                                f"{prefix}/{layer_name}/{component_type}/{subcomp_name}/{param_type}/{stat_name}"
                            )
                            metrics[metric_name] = value

                        # Calculate gradient stats if gradients exist
                        if hasattr(param, "grad") and param.grad is not None:
                            grad_stats = compute_gradient_stats(param)

                            # Add gradient metrics
                            for stat_name, value in grad_stats.items():
                                # Remove 'grad_' prefix for better organization
                                stat_name_short = stat_name.replace("grad_", "")
                                metric_name = f"{prefix}/{layer_name}/{component_type}/{subcomp_name}/{param_type}/grad/{stat_name_short}"
                                metrics[metric_name] = value

    # Get final layer norm with consistent path
    final_ln_path = "model.final_layer_norm.pp_block.weight"
    final_ln = get_attribute_by_path(model, final_ln_path, debug=False)
    if final_ln is not None:
        stats = compute_activation_stats(final_ln)
        for stat_name, value in stats.items():
            metric_name = f"{prefix}/final_layernorm/weight/{stat_name}"
            metrics[metric_name] = value

    return metrics


def collect_accuracy_metrics(
    logits: torch.Tensor, targets: torch.Tensor, prefix: str = "accuracy"
) -> Dict[str, torch.Tensor]:
    """
    Calculate token prediction accuracy metrics.

    Args:
        logits: Model logits [batch_size, seq_len, vocab_size]
        targets: Target tokens [batch_size, seq_len]
        prefix: Metrics name prefix

    Returns:
        Dictionary of accuracy metrics
    """
    metrics = {}

    if logits is None or targets is None:
        if get_metrics_log_level() > 1:
            print("Warning: Cannot compute accuracy metrics - logits or targets is None")
        return metrics

    # Ensure we're working with detached tensors
    logits = logits.detach() if hasattr(logits, "detach") else logits
    targets = targets.detach() if hasattr(targets, "detach") else targets

    # Check shapes
    if logits.dim() < 2 or targets.dim() < 1:
        return metrics

    # Get predictions (indices of max logits)
    predictions = logits.argmax(dim=-1)

    # Calculate accuracy - match shape of targets
    if predictions.shape != targets.shape:
        # Adjust shapes if needed
        if predictions.dim() > targets.dim():
            predictions = predictions[..., -targets.shape[-1] :]
        elif targets.dim() > predictions.dim():
            targets = targets[..., -predictions.shape[-1] :]

    # Calculate overall accuracy
    correct = (predictions == targets).float()
    accuracy = correct.mean()
    metrics[f"{prefix}/overall"] = accuracy

    # Calculate accuracy for next token prediction (last position)
    if targets.dim() > 1 and targets.size(1) > 1:
        next_token_accuracy = correct[:, -1].mean()
        metrics[f"{prefix}/next_token"] = next_token_accuracy

    # Calculate accuracy by position (useful for detecting position bias)
    if targets.dim() > 1 and targets.size(1) > 1:
        for pos in range(min(5, targets.size(1))):
            pos_accuracy = correct[:, pos].mean()
            metrics[f"{prefix}/position_{pos}"] = pos_accuracy

    # Track number of tokens for proper averaging
    metrics[f"{prefix}/num_tokens"] = torch.tensor(targets.numel(), device=targets.device)

    return metrics


def collect_loss_metrics(loss: torch.Tensor, prefix: str = "loss") -> Dict[str, torch.Tensor]:
    """
    Collect metrics related to the loss.

    Args:
        loss: The loss tensor
        prefix: Metrics name prefix

    Returns:
        Dictionary of loss metrics
    """
    metrics = {}

    if loss is None:
        return metrics

    # Ensure we're working with a detached tensor
    loss = loss.detach() if hasattr(loss, "detach") else loss

    # Basic loss metrics
    metrics[f"{prefix}/value"] = loss

    return metrics


def collect_performance_metrics(
    step_time: float, batch_size: int, seq_length: int, prefix: str = "performance"
) -> Dict[str, torch.Tensor]:
    """
    Collect performance metrics like throughput.

    Args:
        step_time: Time taken for a step in seconds
        batch_size: Batch size used
        seq_length: Sequence length used
        prefix: Metrics name prefix

    Returns:
        Dictionary of performance metrics
    """
    metrics = {}

    # Convert to tensors for consistency
    device = "cpu"  # Performance metrics can be on CPU

    # Record step time
    metrics[f"{prefix}/step_time_seconds"] = torch.tensor(step_time, device=device)

    # Record tokens per second
    tokens_per_batch = batch_size * seq_length
    tokens_per_second = tokens_per_batch / step_time if step_time > 0 else 0
    metrics[f"{prefix}/tokens_per_second"] = torch.tensor(tokens_per_second, device=device)

    # Record batch size and sequence length
    metrics[f"{prefix}/batch_size"] = torch.tensor(batch_size, device=device)
    metrics[f"{prefix}/seq_length"] = torch.tensor(seq_length, device=device)

    return metrics


def collect_all_metrics(
    model: torch.nn.Module,
    logits: Optional[torch.Tensor] = None,
    targets: Optional[torch.Tensor] = None,
    loss: Optional[torch.Tensor] = None,
    step_time: Optional[float] = None,
    batch_size: Optional[int] = None,
    seq_length: Optional[int] = None,
    log_level: Optional[int] = None,
    debug: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Collect all metrics based on the specified log level.

    Args:
        model: The model to analyze
        logits: Model output logits
        targets: Target tokens
        loss: Loss value
        step_time: Time taken for a step
        batch_size: Batch size used
        seq_length: Sequence length used
        log_level: Override the log level from environment variable
        debug: Whether to print debug info

    Returns:
        Dictionary of all requested metrics
    """
    if log_level is None:
        log_level = get_metrics_log_level()

    metrics = {}

    # Always collect basic metrics
    if loss is not None:
        metrics.update(collect_loss_metrics(loss))

    if step_time is not None and batch_size is not None and seq_length is not None:
        metrics.update(collect_performance_metrics(step_time, batch_size, seq_length))

    # Collect accuracy metrics if logits and targets are available
    if logits is not None and targets is not None:
        metrics.update(collect_accuracy_metrics(logits, targets))

    # Medium and full detail metrics
    if log_level >= 1:
        # Add embedding metrics
        try:
            metrics.update(collect_embeddings_metrics(model, debug=debug))
        except Exception as e:
            if debug:
                print(f"Error collecting embedding metrics: {e}")

        # Full detail metrics
        if log_level >= 2:
            # Add activation metrics
            try:
                metrics.update(collect_activation_metrics(model, debug=debug))
            except Exception as e:
                if debug:
                    print(f"Error collecting activation metrics: {e}")

            # Add attention metrics
            try:
                metrics.update(collect_attention_metrics(model, debug=debug))
            except Exception as e:
                if debug:
                    print(f"Error collecting attention metrics: {e}")

            # Add parameter metrics
            try:
                metrics.update(collect_parameter_metrics(model, debug=debug))
            except Exception as e:
                if debug:
                    print(f"Error collecting parameter metrics: {e}")

    return metrics


if __name__ == "__main__":
    # Simple test code
    print("Metrics logging utilities loaded")
