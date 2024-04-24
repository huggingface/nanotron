import glob
import os


def generate_needle_in_haystack_test(
    needle, haystack_dir, soft_prompt: str, retrieval_question, context_length, depth_percent
):
    def read_context_files():
        context = ""
        base_dir = os.path.abspath(os.path.dirname(__file__))  # Package directory

        while len(context) < context_length:
            for file in glob.glob(os.path.join(base_dir, haystack_dir, "*.txt")):
                with open(file, "r") as f:
                    context += f.read()
        return context

    def insert_needle(context):
        if depth_percent == 100:
            # If depth percent is 100, place the needle at the end
            new_context = context[: context_length - len(needle)] + needle
        else:
            # Get the position to insert the needle
            insertion_point = int(context_length * (depth_percent / 100))

            # Find the nearest period to the insertion point
            while context[insertion_point] != "." and insertion_point > 0:
                insertion_point -= 1

            # Insert the needle at the appropriate position
            new_context = context[:insertion_point] + needle + context[insertion_point:context_length]

        return new_context

    # Load up the haystack context
    context = read_context_files()

    # Truncate the context to the desired context length
    context = context[:context_length]

    # Insert the needle into the context at the specified depth percent
    context_with_needle = insert_needle(context)

    # Generate the prompt using the context with the needle
    prompt = f"{soft_prompt} {context_with_needle} \n\n{retrieval_question}"

    return prompt, needle


# import numpy as np
# def generate_needle_in_haystack_test(needle, haystack_dir, retrieval_question,
#                                      context_lengths_min=1000, context_lengths_max=16000,
#                                      context_lengths_num_intervals=35, context_lengths=None,
#                                      document_depth_percent_min=0, document_depth_percent_max=100,
#                                      document_depth_percent_intervals=35, document_depth_percents=None,
#                                      document_depth_percent_interval_type="linear"):
#     def read_context_files(context_length):
#         context = ""
#         base_dir = os.path.abspath(os.path.dirname(__file__))  # Package directory

#         while len(context) < context_length:
#             for file in glob.glob(os.path.join(base_dir, haystack_dir, "*.txt")):
#                 with open(file, 'r') as f:
#                     context += f.read()
#         return context

#     def insert_needle(context, depth_percent):
#         if depth_percent == 100:
#             # If depth percent is 100, place the needle at the end
#             new_context = context[:len(context) - len(needle)] + needle
#         else:
#             # Get the position to insert the needle
#             insertion_point = int(len(context) * (depth_percent / 100))

#             # Find the nearest period to the insertion point
#             while context[insertion_point] != '.' and insertion_point > 0:
#                 insertion_point -= 1

#             # Insert the needle at the appropriate position
#             new_context = context[:insertion_point] + needle + context[insertion_point:]

#         return new_context

#     def generate_intervals(min_val, max_val, num_intervals, interval_type):
#         if interval_type == "linear":
#             intervals = np.linspace(min_val, max_val, num_intervals, dtype=int)
#         elif interval_type == "log":
#             intervals = np.logspace(np.log10(min_val), np.log10(max_val), num_intervals, dtype=int)
#         else:
#             raise ValueError("Invalid interval type. Supported types: 'linear' or 'log'.")
#         return intervals

#     if context_lengths is None:
#         context_lengths = generate_intervals(context_lengths_min, context_lengths_max,
#                                              context_lengths_num_intervals, "linear")

#     if document_depth_percents is None:
#         document_depth_percents = generate_intervals(document_depth_percent_min, document_depth_percent_max,
#                                                      document_depth_percent_intervals, document_depth_percent_interval_type)

#     prompts = []
#     for context_length in context_lengths:
#         for depth_percent in document_depth_percents:
#             # Load up the haystack context
#             context = read_context_files(context_length)

#             # Insert the needle into the context at the specified depth percent
#             context_with_needle = insert_needle(context, depth_percent)

#             # Generate the prompt using the context with the needle
#             prompt = f"{context_with_needle}\n\n{retrieval_question}"
#             prompts.append(prompt)

#     return prompts, needle

# # Working example
haystack_dir = "./"
# needle = "The quick brown fox jumps over the lazy dog."
# retrieval_question = "What is the animal mentioned in the text?"
# context_length = 1000
# depth_percent = 80


# # Create the haystack directory if it doesn't exist
# os.makedirs(haystack_dir, exist_ok=True)

# # Create sample text files in the haystack directory
# file_contents = [
#     "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
#     "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
#     "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
#     "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.",
#     "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
# ]

# # Create text files in the haystack directory
# for i, content in enumerate(file_contents, start=1):
#     file_name = f"file{i}.txt"
#     file_path = os.path.join(haystack_dir, file_name)
#     with open(file_path, "w") as file:
#         file.write(content)

# print(f"Haystack directory created at: {haystack_dir}")


# prompt, needle = generate_needle_in_haystack_test(needle, haystack_dir, retrieval_question, context_length, depth_percent)

# print("Generated Prompt:")
# print(prompt)
# print("\nNeedle:")
# print(needle)

import random
import uuid

import numpy as np
from datasets import Dataset


def generate_dataset(max_length=1000):
    # context_lengths = np.logspace(10, np.log10(max_length), num=10, dtype=int)
    soft_prompt = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
    context_lengths = [1024, 2048]
    depth_percents = np.linspace(0, 100, num=21)

    dataset_dict = {"id": [], "prompt": [], "answer": [], "context_length": [], "depth_percent": []}
    generated_ids = set()

    for context_length in context_lengths:
        for depth_percent in depth_percents:
            pass_key = random.randint(1, 10000)
            needle = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key. "
            retrieval_question = "What is the pass key? The pass key is "

            prompt, _ = generate_needle_in_haystack_test(
                needle, haystack_dir, soft_prompt, retrieval_question, context_length, depth_percent
            )
            # answer = f"The pass key is {pass_key}"

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
dataset.save_to_disk("simple_needle_in_a_hay_stack")
dataset.push_to_hub("nanotron/simple_needle_in_a_hay_stack")
