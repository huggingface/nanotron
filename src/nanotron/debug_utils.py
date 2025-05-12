import logging
from typing import Optional

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

def debug_dataloader_samples(
    dataloader: DataLoader,
    tokenizer: PreTrainedTokenizer,
    num_samples: int = 2
) -> None:
    """
    Debug utility to inspect samples from a DataLoader.
    
    This function pulls the first batch from the given DataLoader,
    detokenizes the 'input_ids' using the provided tokenizer,
    and prints the decoded texts for a few samples.
    
    Args:
        dataloader (torch.utils.data.DataLoader): The DataLoader to inspect.
        tokenizer (PreTrainedTokenizer): Tokenizer used to decode input_ids.
        num_samples (int): Number of samples to print from the first batch.        
    """
    try: 
        batch = next(iter(dataloader))
    except Exception as e:
        logger.error("[debug] Failed to retrieve batch from dataloader: %s", e)
        return
    
    input_ids = batch.get("input_ids")
    if input_ids is None:
        logger.warning("[debug] 'input_ids' not found in batch. Available keys: %s", list(batch.keys()))
        return
    
    if hasattr(input_ids, "cpu"):
        input_ids = input_ids.cpu()
    
    logger.info("\n[Debug] Printing detokenized samples from the first batch:\n")
    for i in range(min(num_samples, len(input_ids))):
        try:
            decoded = tokenizer.decode(input_ids[i], skip_special_tokens=True)
            logger.info("[Sample %d]:\n%s\n%s", i+1, decoded, "=" * 40)
        except Exception as e:
            logger.error("[debug] Failed to decode sample %d: %s", i+1, e)