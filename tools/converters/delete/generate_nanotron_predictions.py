"""
torchrun --nproc-per-node 1 tools/converters/delete/generate_nanotron_predictions.py --tp 1 --nanotron-checkpoint-path /capstor/scratch/cscs/asolergi/nanotron/checkpoints/nanotron_pretrained_checkpoints/Nanotron-Llama-3.2-3B
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

TXT="Paris! Paris is the capital and most populous city of France, located in the north-central part of the country. It is a global center for art, fashion, cuisine, culture, and romance. Here's a brief overview: **History and Culture:**Paris has a rich history dating back to the 3rd century, with a blend of Roman, Gothic, Renaissance, and Art Nouveau influences. The city is famous for its iconic landmarks like the Eiffel Tower (built for the 1889 World's Fair), the Louvre Museum (home to the Mona Lisa), Notre-Dame Cathedral, and the Arc de Triomphe. **Art and Architecture:**Paris is renowned for its stunning architecture, with many beautiful bridges, gardens, and buildings. The city is also a hub for art, with numerous museums, galleries, and street performers. The Louvre, Musée d'Orsay, and Centre Pompidou are just a few of the many world-class museums. **Fashion and Cuisine:**Paris is considered the fashion capital of the world, with top designers like Chanel, Dior, and Louis Vuitton. The city is also famous for its exquisite cuisine, with popular dishes like escargots, croissants, baguettes, and cheese. Don't forget to try a classic French dessert like crème brûlée or macarons! **Romance and Entertainment:**Paris is often called the City of Light (La Ville Lumière) and the City of Love. It's a popular destination for couples and honeymooners, with its picturesque Seine River, charming streets, and cozy cafes. The city also hosts many festivals and events, including the French Open tennis tournament, the Tour de France, and the Rock en Seine music festival. **Economy and Education:** Paris is a global economic hub, with many multinational companies, startups, and universities. The city is home to some of the world's top universities, including the Sorbonne and École des Hautes Études en Sciences Sociales (EHESS). **Tourism:** Paris is one of the most visited cities in the world, attracting over 23 million tourists annually. Visitors come to experience the city's unique blend of history, culture, art, fashion, and romance. In summary, Paris is a vibrant, elegant, and enchanting city that offers something for everyone: history, art, fashion, cuisine, romance, and entertainment."
SEQ_LENGTH = 256  # For truncating the TXT if GPU can't fit too many tokens

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
        predictions = np.argmax(output.transpose(0, 1).to(torch.float).cpu(), axis=2).flatten().tolist()
        labels = tokens.cpu().flatten()[1:].tolist()
        print(f"\nAccuracy: {accuracy_score(labels, predictions)}")

if __name__ == "__main__":
    _args = get_args()
    main(_args)
