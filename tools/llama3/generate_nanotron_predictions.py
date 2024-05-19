"""
torchrun --nproc-per-node 2 tools/llama3/generate_nanotron_predictions.py --tp 2 --nanotron-checkpoint-path nanotron_checkpoints/NanotronLlama38B
"""
import argparse
import os
from pathlib import Path

import nanotron.distributed as dist
import numpy as np
import torch
from nanotron.config import Config, ParallelismArgs, get_config_from_file
from nanotron.models import build_model
from nanotron.models.llama import LlamaForTraining
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import sanity_check
from nanotron.parallel.pipeline_parallel.engine import AllForwardAllBackwardPipelineEngine
from nanotron.parallel.tensor_parallel.nn import TensorParallelLinearMode
from nanotron.serialize import load_weights
from nanotron.trainer import mark_tied_parameters
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer

TXT = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHello! Which is the capital of France? What can I visit over there if I go for a week vacation?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nBonjour! The capital of France is Paris, also known as the City of Light. Paris is a stunning city with a rich history, art, fashion, and cuisine. If you're planning a week-long vacation in Paris, you'll have plenty of time to explore its iconic landmarks, museums, and neighborhoods. Here's a suggested itinerary to get you started:  Day 1-2: Iconic Landmarks  The Eiffel Tower (Tour Eiffel): The iron lady offers breathtaking views of the city. You can take the stairs or elevator to the top. The Louvre Museum (Musée du Louvre): Home to the Mona Lisa, Venus de Milo, and many other famous artworks. Arc de Triomphe: A monumental arch honoring the soldiers who fought and died for France. Champs-Élysées: A famous avenue lined with cafes, shops, and theaters. Day 3: Montmartre and Sacré-Cœur  Explore the charming neighborhood of Montmartre, known for its bohemian vibe, street artists, and stunning views. Visit the Basilique du Sacré-Cœur, a beautiful white church perched on a hill."
SEQ_LENGTH = 512  # For truncating the TXT if GPU can't fit too many tokens

DEVICE = torch.device("cuda")
TORCH_DTYPE = torch.bfloat16


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="Nanotron Model")
    group.add_argument(
        "--nanotron-checkpoint-path",
        type=str,
        required=True,
        help="A path to a directory containing a Nanotron Checkpoint",
    )

    group = parser.add_argument_group(title="Nanotron Parallelism")
    group.add_argument("--tp", type=int, required=True, help="Tensor Parallelism Degree of the Nanotron Checkpoint")

    args = parser.parse_args()

    return args


def main(args):
    # Init Nanotron Parallel Utilities
    parallel_config = ParallelismArgs(
        dp=1,
        pp=1,
        tp=args.tp,
        pp_engine=AllForwardAllBackwardPipelineEngine(),
        tp_mode=TensorParallelLinearMode.ALL_REDUCE,
        tp_linear_async_communication=False,
    )
    assert (
        parallel_config.tp_mode == TensorParallelLinearMode.ALL_REDUCE
        and parallel_config.tp_linear_async_communication is False
    )

    parallel_context = ParallelContext(
        data_parallel_size=parallel_config.dp,
        pipeline_parallel_size=parallel_config.pp,
        tensor_parallel_size=parallel_config.tp,
    )

    RANK = dist.get_rank(parallel_context.world_pg)

    nanotron_config = get_config_from_file(
        os.path.join(args.nanotron_checkpoint_path, "config.yaml"), config_class=Config, model_config_class=None
    )

    model = build_model(
        model_builder=lambda: LlamaForTraining(
            config=nanotron_config.model.model_config,
            parallel_context=parallel_context,
            parallel_config=parallel_config,
            random_states=None,
        ),
        parallel_context=parallel_context,
        dtype=TORCH_DTYPE,
        device=DEVICE,  # TODO Check with different parallelism if cpu is available
    )

    mark_tied_parameters(model=model, parallel_context=parallel_context)
    sanity_check(root_module=model)

    # Load checkpoint directly in memory and then only keep the state dictionary
    load_weights(model=model, parallel_context=parallel_context, root_folder=Path(args.nanotron_checkpoint_path))

    tokenizer = AutoTokenizer.from_pretrained(nanotron_config.tokenizer.tokenizer_name_or_path)
    tokens = tokenizer(TXT, return_tensors="pt", truncation=True, max_length=(SEQ_LENGTH + 1))["input_ids"].to(DEVICE)
    inputs = {"input_ids": tokens[:, :-1], "input_mask": torch.ones((1, SEQ_LENGTH), device=DEVICE)}

    model.eval()

    with torch.no_grad():
        output = model.model(**inputs)

    if not RANK:
        predicted_tokens = [5, 27, 34]  # Index of the predictions to compare across models
        term_cols = int(os.get_terminal_size().columns / 3)

        for predicted_token in predicted_tokens:

            print("\n", "=" * term_cols, f"Predictions of token {predicted_token}", "=" * term_cols)
            next_tokens = torch.softmax(output.transpose(0, 1)[0, predicted_token, :], -1)
            topk_next_tokens = torch.topk(next_tokens, 10)

            print(
                *[
                    f"[Nanotron Model] Next token: {idx.item()}, probability: {prob}"
                    for idx, prob in zip(topk_next_tokens.indices, topk_next_tokens.values)
                ],
                sep="\n",
            )

        # Compute accuracy
        predictions = np.argmax(output.transpose(0, 1).cpu(), axis=2).flatten().tolist()
        labels = tokens.cpu().flatten()[1:].tolist()
        print(f"\nAccuracy: {accuracy_score(labels, predictions)}")
        # Results
        ## Nanotron 8B, TP 1: 0.8272058823529411
        ## Nanotron 8B, TP 2: 0.7720588235294118
        ## Nanotron 70B, TP 2: 0.8272058823529411


if __name__ == "__main__":
    _args = get_args()
    main(_args)
