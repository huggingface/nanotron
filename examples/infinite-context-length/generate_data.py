import argparse
import glob
import os
import random
import uuid

from datasets import Dataset
from transformers import AutoTokenizer

PROMPT = "{} {}. \n\n{}"


def token_length(tokenizer, text):
    # Exclude EOS token
    return len(tokenizer.encode(text))


def read_context_files(tokenizer, soft_prompt, retrieval_question, target_cut_length):
    context = ""
    base_dir = os.path.abspath(os.path.dirname(__file__))
    files = glob.glob(os.path.join(base_dir, haystack_dir, "*.txt"))
    file_index = 0

    # target_context_length = context_length - token_length(tokenizer, soft_prompt) - token_length(tokenizer, retrieval_question)

    # while token_length(tokenizer, f"{soft_prompt} {context}. \n\n{retrieval_question}") < target_cut_length:
    while token_length(tokenizer, PROMPT.format(soft_prompt, context, retrieval_question)) < target_cut_length:
        with open(files[file_index], "r") as f:
            content = f.read()
            # Ensure the token length of the context does not exceed the target token length
            # if token_length(tokenizer, f"{soft_prompt} {context + content}. \n\n{retrieval_question}") > target_cut_length:
            if (
                token_length(tokenizer, PROMPT.format(soft_prompt, context + content, retrieval_question))
                > target_cut_length
            ):
                # truncated_content = content[:-(target_cut_length - token_length(tokenizer, f"{soft_prompt} {context}. \n\n{retrieval_question}"))]
                truncated_content = content[
                    : -(
                        target_cut_length
                        - token_length(tokenizer, PROMPT.format(soft_prompt, context, retrieval_question))
                    )
                ]
                context += truncated_content
            else:
                context += content
        file_index = (file_index + 1) % len(files)

    return context


# def insert_needle(context, needle):
#     # Get the position to insert the needle
#     insertion_point = random.randint(0, len(context) - len(needle)) - 1

#     # Insert the needle at the appropriate position
#     new_context = context[:insertion_point] + needle + context[insertion_point:]

#     return new_context


def insert_needle_with_depth(needle, context, depth_percent, target_cut_length, tokenizer):
    content_ids = tokenizer.encode(context)
    # content_length = len(content_ids)
    needle_ids = tokenizer.encode(needle)

    if depth_percent == 100:
        # If depth percent is 100, place the needle at the end
        # new_context = context[: len(context) - len(needle)] + needle
        new_context_ids = content_ids[: len(content_ids) - len(needle_ids)] + needle_ids
    else:
        # Get the position to insert the needle
        # insertion_point = int(context_length * (depth_percent / 100))
        insertion_point = int(len(content_ids) * (depth_percent / 100))

        # Find the nearest period to the insertion point
        while context[insertion_point] != "." and insertion_point > 0:
            insertion_point -= 1

        # Insert the needle at the appropriate position
        # new_context = context[:insertion_point] + needle + context[insertion_point:content_length]
        new_context_ids = (
            content_ids[:insertion_point]
            + needle_ids
            + content_ids[insertion_point : (target_cut_length - len(needle_ids))]
        )

    new_content = tokenizer.decode(new_context_ids)

    return new_content


def generate_needle_in_haystack_test(
    needle, needle_prompt, haystack_dir, soft_prompt: str, retrieval_question, context_length, depth_percent
):
    # Load up the haystack context
    tokenizer = AutoTokenizer.from_pretrained("lvwerra/the-tokenizer-v1")
    target_cut_length = context_length - token_length(tokenizer, PROMPT.format(soft_prompt, 1, retrieval_question)) - 1

    context = read_context_files(tokenizer, soft_prompt, retrieval_question, target_cut_length)

    # Insert the needle into the context at the specified depth percent
    context_with_needle = insert_needle_with_depth(needle_prompt, context, depth_percent, target_cut_length, tokenizer)

    # Generate the prompt using the context with the needle
    prompt = f"{soft_prompt} {context_with_needle}. \n\n{retrieval_question}"

    assert str(needle) in context_with_needle, f"depth_percent: {depth_percent}"
    # assert abs(context_length - token_length(tokenizer, prompt)) <= 10
    # assert context_length - token_length(tokenizer, prompt)

    # remaining_tokens = context_length - token_length(tokenizer, prompt)
    # NOTE: now add `.` to soft_prompt so that the token length is exactly equal to context_length
    while (
        token_length(tokenizer, PROMPT.format(soft_prompt, context_with_needle, retrieval_question)) < context_length
    ):
        soft_prompt += "."

    prompt = PROMPT.format(soft_prompt, context_with_needle, retrieval_question)
    assert (
        token_length(tokenizer, prompt) == context_length
    ), f"Token length: {token_length(tokenizer, prompt)}, Context length: {context_length}, depth_percent: {depth_percent}"

    return prompt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context_length", type=int, required=True)
    parser.add_argument("--depth_percent", type=int, required=True)
    parser.add_argument("--id", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    context_length = args.context_length
    depth_percent = args.depth_percent
    id = args.id

    haystack_dir = "./"
    # NOTE: depth_percent + 1 to avoid 0
    start_range = 1000 * (depth_percent + 1) * id
    end_range = start_range + start_range

    print(
        f"Generating prompts for context length: {context_length} and depth percent: {depth_percent} and id: {id} \n"
    )
    print(f"start_range: {start_range}, end_range: {end_range} \n")

    def generate_dataset():
        # num_prompts = 1700
        num_prompts = 100
        # soft_prompt = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
        soft_prompt = "There is a pass key hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about what is the pass key later on."
        # context_lengths = [
        #     32768,
        # ]
        # depth_percents = np.linspace(0, 100, num=21)

        dataset_dict = {"id": [], "prompt": [], "answer": [], "context_length": [], "depth_percent": []}
        generated_ids = set()
        generated_pass_keys = set()

        # for context_length in context_lengths:
        #     print(f"Generating prompts for context length: {context_length} \n")
        #     for depth_percent in depth_percents:
        for i in range(num_prompts):
            print(f"generating prompt {i} \n")

            while True:
                pass_key = random.randint(start_range, end_range)
                if pass_key not in generated_pass_keys:
                    generated_pass_keys.add(pass_key)
                    break

            needle_prompt = f". The pass key is {pass_key}. Remember it. {pass_key} is the pass key. "
            retrieval_question = f"What is the pass key? The pass key is {pass_key}."

            prompt = generate_needle_in_haystack_test(
                pass_key,
                needle_prompt,
                haystack_dir,
                soft_prompt,
                retrieval_question,
                context_length,
                depth_percent,
            )

            while True:
                text_id = str(uuid.uuid4())
                if text_id not in generated_ids:
                    generated_ids.add(text_id)
                    break

            dataset_dict["id"].append(text_id)
            dataset_dict["prompt"].append(prompt)
            dataset_dict["answer"].append(pass_key)
            dataset_dict["context_length"].append(context_length)
            dataset_dict["depth_percent"].append(depth_percent)

        dataset = Dataset.from_dict(dataset_dict)
        return dataset

    # Generate the dataset
    dataset = generate_dataset()

    # Save the dataset to disk
    dataset.save_to_disk(
        f"/fsx/phuc/projects/nanotron/examples/infinite-context-length/needle_finetune_datasets/needle_finetuning_ctx_len_32768_and_depth_{depth_percent}_and_id_{id}"
    )
    # dataset.push_to_hub("nanotron/needle_in_a_hay_stack_finetuning_dataset")
