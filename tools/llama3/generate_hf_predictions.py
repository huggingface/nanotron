"""
torchrun --nproc-per-node 1 tools/llama3/generate_hf_predictions.py --pretrained-model-name-or-path meta-llama/Meta-Llama-3-8B-Instruct
"""
import argparse
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer

TXT = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHello! Which is the capital of France? What can I visit over there if I go for a week vacation?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nBonjour! The capital of France is Paris, also known as the City of Light. Paris is a stunning city with a rich history, art, fashion, and cuisine. If you're planning a week-long vacation in Paris, you'll have plenty of time to explore its iconic landmarks, museums, and neighborhoods. Here's a suggested itinerary to get you started:  Day 1-2: Iconic Landmarks  The Eiffel Tower (Tour Eiffel): The iron lady offers breathtaking views of the city. You can take the stairs or elevator to the top. The Louvre Museum (Musée du Louvre): Home to the Mona Lisa, Venus de Milo, and many other famous artworks. Arc de Triomphe: A monumental arch honoring the soldiers who fought and died for France. Champs-Élysées: A famous avenue lined with cafes, shops, and theaters. Day 3: Montmartre and Sacré-Cœur  Explore the charming neighborhood of Montmartre, known for its bohemian vibe, street artists, and stunning views. Visit the Basilique du Sacré-Cœur, a beautiful white church perched on a hill."
SEQ_LENGTH = 512  # For truncating the TXT if GPU can't fit too many tokens

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
        device_map="auto",
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

    # Compute accuracy
    predictions = np.argmax(output.logits.cpu(), axis=2).flatten().tolist()
    labels = tokens.cpu().flatten()[1:].tolist()
    print(f"\nAccuracy: {accuracy_score(labels, predictions)}")
    # Results
    ## [TP=1] HF 8B: 0.8308823529411765
    ## [TP=2]HF 70B: 0.8860294117647058
    ## [TP=1] HF -> Nanotron -> HF 8B: 0.8308823529411765
    ## [TP=2] HF -> Nanotron -> HF 70B: 0.8860294117647058
    ## [TP=1 --> TP=2] HF -> Nanotron -> Dummy Finetune to change TP=2 -> HF 8B: 0.8308823529411765


if __name__ == "__main__":
    _args = get_args()
    main(_args)
