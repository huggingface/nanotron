import argparse
import glob
import os
import random
import uuid

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

PROMPT = "{} {}. \n\n{}"


def get_keys_in_train_set(dataset):
    unique_answers = set()
    for split in dataset.keys():
        for example in dataset[split]:
            answer = example["answer"]
            unique_answers.add(answer)

    return unique_answers


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


def insert_needle_with_depth(needle, context, depth_percent, target_cut_length, tokenizer):
    content_ids = tokenizer.encode(context)
    needle_ids = tokenizer.encode(needle)

    if depth_percent == 100:
        # If depth percent is 100, place the needle at the end
        new_context_ids = content_ids[: len(content_ids) - len(needle_ids)] + needle_ids
    else:
        # Get the position to insert the needle
        insertion_point = int(len(content_ids) * (depth_percent / 100))

        # Find the nearest period to the insertion point
        while context[insertion_point] != "." and insertion_point > 0:
            insertion_point -= 1

        # new_context = context[:insertion_point] + needle + context[insertion_point:content_length]
        new_context_ids = (
            content_ids[:insertion_point]
            + needle_ids
            + content_ids[insertion_point : (target_cut_length - len(needle_ids))]
        )

    new_content = tokenizer.decode(new_context_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    return new_content


def generate_needle_in_haystack_test(
    needle,
    needle_prompt,
    soft_prompt: str,
    retrieval_question,
    context_length,
    depth_percent,
    tokenizer,
    is_padding: bool,
):
    target_cut_length = context_length - token_length(tokenizer, PROMPT.format(soft_prompt, 1, retrieval_question)) - 1

    context = read_context_files(tokenizer, soft_prompt, retrieval_question, target_cut_length)

    # Insert the needle into the context at the specified depth percent
    context_with_needle = insert_needle_with_depth(needle_prompt, context, depth_percent, target_cut_length, tokenizer)

    # Generate the prompt using the context with the needle
    prompt = f"{soft_prompt} {context_with_needle}. \n\n{retrieval_question}"

    assert str(needle) in context_with_needle, f"depth_percent: {depth_percent}"

    # NOTE: now add `.` to soft_prompt so that the token length is exactly equal to context_length
    if is_padding is True:
        while (
            token_length(tokenizer, PROMPT.format(soft_prompt, context_with_needle, retrieval_question))
            < context_length
        ):
            soft_prompt += "."

    prompt = PROMPT.format(soft_prompt, context_with_needle, retrieval_question)

    if is_padding is True:
        assert (
            token_length(tokenizer, prompt) == context_length
        ), f"Token length: {token_length(tokenizer, prompt)}, Context length: {context_length}, depth_percent: {depth_percent}"

    return prompt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context_length", type=int, required=True)
    parser.add_argument("--depth_percent", type=int, required=True)
    parser.add_argument("--num_prompts", type=int, default=5)
    parser.add_argument(
        "--tokenizer_path", type=str, default="/fsx/haojun/lighteval_evaluation_model/NanotronLlama3-8B"
    )
    parser.add_argument("--id", type=int, required=True)
    parser.add_argument("--haystack_dir", type=str, default="./haystack_txt/")
    parser.add_argument("--is_push_to_hub", type=bool, default=False)
    parser.add_argument("--is_exact_context_length", type=int, default=1)  # 1 is True, 0 is False
    parser.add_argument("--is_padding", type=int, default=1)  # 1 is True, 0 is False
    parser.add_argument("--is_eval", type=int, default=1)  # 1 is True, 0 is False
    parser.add_argument("--check_key_in_dataset", type=str, default=None)  # 1 is True, 0 is False
    parser.add_argument("--save_path", type=str, required=True)  # 1 is True, 0 is False
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    for key, value in vars(args).items():
        print(f"{key}: {value}")

    context_length = args.context_length
    depth_percent = args.depth_percent
    tokenizer_path = args.tokenizer_path
    num_prompts = args.num_prompts
    id = args.id
    haystack_dir = args.haystack_dir
    is_push_to_hub = args.is_push_to_hub
    # is_exact_context_length = args.is_exact_context_length
    is_exact_context_length = False if args.is_exact_context_length == 0 else True
    is_padding = False if args.is_padding == 0 else True
    is_eval = False if args.is_eval == 0 else True
    check_key_in_dataset = args.check_key_in_dataset
    save_path = args.save_path

    assert save_path is not None

    if check_key_in_dataset is not None:
        eval_dataset = load_dataset(check_key_in_dataset)
        eval_keys = get_keys_in_train_set(eval_dataset)

    if context_length >= 2000:
        # NOTE: don't minus short context
        gen_context_length = (
            context_length if is_exact_context_length is True else context_length - random.randint(0, 700)
        )
    else:
        gen_context_length = context_length

    # NOTE: depth_percent + 1 to avoid 0
    RANGE = 500
    start_range = 30 * (depth_percent + 1) * id
    end_range = start_range + RANGE

    print(
        f"Generating prompts for context length: {gen_context_length} (original {context_length}) and depth percent: {depth_percent} and id: {id} \n"
    )
    print(f"start_range: {start_range}, end_range: {end_range} \n")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def generate_dataset():
        soft_prompt = "There is a pass key hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about what is the pass key later on."

        dataset_dict = {
            "id": [],
            "prompt": [],
            "answer": [],
            "context_length": [],
            "num_tokens": [],
            "depth_percent": [],
        }
        generated_ids = set()
        generated_pass_keys = set()

        for i in range(num_prompts):
            print(f"generating prompt {i} \n")

            while True:
                pass_key = random.randint(start_range, end_range)
                if pass_key not in generated_pass_keys:
                    if check_key_in_dataset is not None:
                        if str(pass_key) in eval_keys:
                            continue

                    generated_pass_keys.add(pass_key)
                    break

            needle_prompt = f". The pass key is {pass_key}. Remember it. {pass_key} is the pass key. "

            if is_eval is True:
                retrieval_question = "What is the pass key? The pass key is "
            else:
                retrieval_question = f"What is the pass key? The pass key is {pass_key}."

            prompt = generate_needle_in_haystack_test(
                needle=pass_key,
                needle_prompt=needle_prompt,
                soft_prompt=soft_prompt,
                retrieval_question=retrieval_question,
                context_length=gen_context_length,
                depth_percent=depth_percent,
                tokenizer=tokenizer,
                is_padding=is_padding,
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
            dataset_dict["num_tokens"].append(token_length(tokenizer, prompt))
            dataset_dict["depth_percent"].append(depth_percent)

        dataset = Dataset.from_dict(dataset_dict)
        return dataset

    # Generate the dataset
    dataset = generate_dataset()

    # Save the dataset to disk
    dataset.save_to_disk(
        f"{save_path}/needle_finetune_data_and_{context_length}_ctx_and_depth_{depth_percent}_and_id_{id}"
    )
    # if is_push_to_hub:
    #     dataset.push_to_hub("nanotron/llama3-16k-passkey-retrieval-eval")
