"""
torchrun --nproc-per-node 1 tools/converters/delete/generate_hf_predictions.py --pretrained-model-name-or-path meta-llama/Llama-3.2-3B
"""
import argparse
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer

TXT="Paris! Paris is the capital and most populous city of France, located in the north-central part of the country. It is a global center for art, fashion, cuisine, culture, and romance. Here's a brief overview: **History and Culture:**Paris has a rich history dating back to the 3rd century, with a blend of Roman, Gothic, Renaissance, and Art Nouveau influences. The city is famous for its iconic landmarks like the Eiffel Tower (built for the 1889 World's Fair), the Louvre Museum (home to the Mona Lisa), Notre-Dame Cathedral, and the Arc de Triomphe. **Art and Architecture:**Paris is renowned for its stunning architecture, with many beautiful bridges, gardens, and buildings. The city is also a hub for art, with numerous museums, galleries, and street performers. The Louvre, Musée d'Orsay, and Centre Pompidou are just a few of the many world-class museums. **Fashion and Cuisine:**Paris is considered the fashion capital of the world, with top designers like Chanel, Dior, and Louis Vuitton. The city is also famous for its exquisite cuisine, with popular dishes like escargots, croissants, baguettes, and cheese. Don't forget to try a classic French dessert like crème brûlée or macarons! **Romance and Entertainment:**Paris is often called the City of Light (La Ville Lumière) and the City of Love. It's a popular destination for couples and honeymooners, with its picturesque Seine River, charming streets, and cozy cafes. The city also hosts many festivals and events, including the French Open tennis tournament, the Tour de France, and the Rock en Seine music festival. **Economy and Education:** Paris is a global economic hub, with many multinational companies, startups, and universities. The city is home to some of the world's top universities, including the Sorbonne and École des Hautes Études en Sciences Sociales (EHESS). **Tourism:** Paris is one of the most visited cities in the world, attracting over 23 million tourists annually. Visitors come to experience the city's unique blend of history, culture, art, fashion, and romance. In summary, Paris is a vibrant, elegant, and enchanting city that offers something for everyone: history, art, fashion, cuisine, romance, and entertainment."
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

    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=TORCH_DTYPE,
        attn_implementation="flash_attention_2",
    ).to(DEVICE).eval()

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

    # Compute accuracy
    predictions = np.argmax(output.logits.to(torch.float).cpu(), axis=2).flatten().tolist()
    labels = tokens.cpu().flatten()[1:].tolist()
    print(f"\nAccuracy: {accuracy_score(labels, predictions)}")

if __name__ == "__main__":
    _args = get_args()
    main(_args)
