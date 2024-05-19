"""
torchrun --nproc-per-node 1 tools/llama3/generate_hf_predictions.py --pretrained-model-name-or-path hf_checkpoints/ConvertedNanotronLlama38B
"""
import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TXT = "The prologue of Romeo and Juliet calls the title characters “star-crossed lovers”—and the stars do seem to conspire against these young lovers.  Romeo is a Montague, and Juliet a Capulet. Their families are enmeshed in a feud, but the moment they meet—when Romeo and his friends attend a party at Juliets house in disguise—the two fall in love and quickly decide that they want to be married.  A friar secretly marries them, hoping to end the feud. Romeo and his companions almost immediately encounter Juliets cousin Tybalt, who challenges Romeo. When Romeo refuses to fight, Romeos friend Mercutio accepts the challenge and is killed. Romeo then kills Tybalt and is banished. He spends that night with Juliet and then leaves for Mantua.  Juliets father forces her into a marriage with Count Paris. To avoid this marriage, Juliet takes a potion, given her by the friar, that makes her appear dead. The friar will send Romeo word to be at her family tomb when she awakes. The plan goes awry, and Romeo learns instead that she is dead. In the tomb, Romeo kills himself. Juliet wakes, sees his body, and commits suicide. Their deaths appear finally to end the feud."
SEQ_LENGTH = 256  # For truncating the TXT if GPU can't fit too many tokens

DEVICE = torch.device("cuda")
TORCH_DTYPE = torch.bfloat16


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="HuggingFace Model")
    group.add_argument(
        "--pretrained-model-name-or-path",
        type=str,
        required=True,
        help="A path to a directory containing model weights saved using save_pretrained() or the model id of a pretrained model hosted inside a model repo on the Hugging Face Hub",
    )

    args = parser.parse_args()

    return args


def main(args):
    # TODO Refractor with HF pipeline or .generate()?
    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=TORCH_DTYPE,
        attn_implementation="flash_attention_2",
        device=DEVICE,
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    tokens = tokenizer(TXT, return_tensors="pt", truncation=True, max_length=(SEQ_LENGTH + 1))["input_ids"].to(DEVICE)
    inputs = tokens[:, :-1]

    with torch.no_grad():
        output = model(inputs)

    predicted_tokens = [5, 27, 34]  # Index of the predictions to compare across models
    term_cols = int(os.get_terminal_size().columns / 3)

    for predicted_token in predicted_tokens:

        print("\n", "=" * term_cols, f"Predictions of token {predicted_token}", "=" * term_cols)
        next_tokens = torch.softmax(output.logits[0, predicted_token, :], -1)
        topk_next_tokens = torch.topk(next_tokens, 10)

        print(
            *[
                f"[HF Model] Next token: {idx.item()}, probability: {prob}"
                for idx, prob in zip(topk_next_tokens.indices, topk_next_tokens.values)
            ],
            sep="\n",
        )


if __name__ == "__main__":
    _args = get_args()
    main(_args)
