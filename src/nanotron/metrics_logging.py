"""
Utility functions for computing and logging various model metrics.
"""

import os
from typing import Any, Dict, List, Optional

import torch


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


def get_attribute_by_path(obj: Any, path: str) -> Optional[Any]:
    """
    Get an attribute from an object by a dot-separated path.
    Also handles special operations like .H and .detach() at the end of the path.

    Args:
        obj: The object to get attributes from
        path: A dot-separated path to the attribute, can include operations

    Returns:
        The attribute if found, None otherwise
    """
    # Split the path into attribute components and possible operations
    # First handle special operations that might be at the end
    operations = []
    path_parts = path.split(".")

    # Check for operations at the end (like .detach() or .H)
    while path_parts and path_parts[-1] in ["detach()", "detach", "H"]:
        op = path_parts.pop()
        operations.append(op.rstrip("()"))  # Remove parentheses if present

    # Navigate through the object attributes
    current = obj
    for attr in path_parts:
        if hasattr(current, attr):
            current = getattr(current, attr)
        else:
            return None

    # Apply any operations that were found
    for op in operations:
        if op == "detach":
            if hasattr(current, "detach"):
                current = current.detach()
        elif op == "H":  # Hermitian transpose
            if hasattr(current, "H"):
                current = current.H

    return current


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


def compute_activation_stats(activation: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Compute statistics about an activation tensor.

    Args:
        activation (torch.Tensor): The activation tensor to analyze

    Returns:
        Dict containing statistics:
            - mean: Mean activation value
            - std: Standard deviation
            - min: Minimum value
            - max: Maximum value
            - norm: L2 norm
            - abs_mean: Mean of absolute values
            - zero_frac: Fraction of zero/near-zero values
    """
    # Handle empty tensor
    if activation.numel() == 0:
        zero_stats = {
            "mean": torch.tensor(0.0),
            "std": torch.tensor(0.0),
            "min": torch.tensor(0.0),
            "max": torch.tensor(0.0),
            "norm": torch.tensor(0.0),
            "abs_mean": torch.tensor(0.0),
            "zero_frac": torch.tensor(1.0),
        }
        return zero_stats

    # Calculate basic statistics
    mean = activation.mean()
    std = activation.std()
    min_val = activation.min()
    max_val = activation.max()
    norm = compute_tensor_norm(activation)
    abs_mean = activation.abs().mean()

    # Calculate fraction of values close to zero (within 1e-10)
    zero_fraction = (activation.abs() < 1e-10).float().mean()

    return {
        "mean": mean,
        "std": std,
        "min": min_val,
        "max": max_val,
        "norm": norm,
        "abs_mean": abs_mean,
        "zero_frac": zero_fraction,
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


# Configure logging level from environment variable
def get_metrics_log_level():
    """Get the log level from the FULL_LOGS environment variable."""
    full_logs = os.environ.get("FULL_LOGS", "0")
    if full_logs == "2":
        return "full"
    elif full_logs == "1":
        return "medium"
    else:
        return "minimal"


if __name__ == "__main__":
    # Simple test code
    print("Metrics logging utilities loaded")
