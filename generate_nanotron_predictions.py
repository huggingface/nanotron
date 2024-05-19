import argparse
import os

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
from transformers import AutoTokenizer

# TODO Currentyly just sopporting Llama8B that doesn't needs any kind of parallelism
DP = 1
PP = 1

TXT = "The prologue of Romeo and Juliet calls the title characters “star-crossed lovers”—and the stars do seem to conspire against these young lovers.  Romeo is a Montague, and Juliet a Capulet. Their families are enmeshed in a feud, but the moment they meet—when Romeo and his friends attend a party at Juliets house in disguise—the two fall in love and quickly decide that they want to be married.  A friar secretly marries them, hoping to end the feud. Romeo and his companions almost immediately encounter Juliets cousin Tybalt, who challenges Romeo. When Romeo refuses to fight, Romeos friend Mercutio accepts the challenge and is killed. Romeo then kills Tybalt and is banished. He spends that night with Juliet and then leaves for Mantua.  Juliets father forces her into a marriage with Count Paris. To avoid this marriage, Juliet takes a potion, given her by the friar, that makes her appear dead. The friar will send Romeo word to be at her family tomb when she awakes. The plan goes awry, and Romeo learns instead that she is dead. In the tomb, Romeo kills himself. Juliet wakes, sees his body, and commits suicide. Their deaths appear finally to end the feud."
SEQ_LENGTH = 256  # For truncating the TXT if GPU can't fit too many tokens


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

    group = parser.add_argument_group(title="Tokenizer")
    group.add_argument(
        "--tokenizer-name-or-path",
        type=str,
        required=True,
        help="A path to a directory containing vocabulary files required by the tokenizer or the model id of a predefined tokenizer hosted inside a model repo on the Hugging Face Hub.",
    )

    args = parser.parse_args()

    return args


def main(args):

    parallel_config = ParallelismArgs(
        dp=DP,
        pp=PP,
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
        dtype=torch.bfloat16,
        device=torch.device("cuda"),  # TODO Check with different parallelism
    )

    mark_tied_parameters(model=model, parallel_context=parallel_context)
    sanity_check(root_module=model)

    # Load checkpoint directly in memory and then only keep the state dictionary
    load_weights(model=model, parallel_context=parallel_context, root_folder=args.nanotron_checkpoint_path)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    tokens = tokenizer(TXT, return_tensors="pt", truncation=True, max_length=(SEQ_LENGTH + 1))["input_ids"].to("cuda")
    inputs = {"input_ids": tokens[:, :-1], "input_mask": torch.ones((1, SEQ_LENGTH), device="cuda")}

    model.eval()

    with torch.no_grad():
        output = model(inputs)

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


if __name__ == "__main__":
    _args = get_args()
    main(_args)
