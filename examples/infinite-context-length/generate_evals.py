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


# # Working example
haystack_dir = "./"

import random
import uuid

import numpy as np
from datasets import Dataset


def generate_dataset():
    num_prompts = 100
    soft_prompt = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
    context_lengths = [
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        # 65536,
        # 131072,
        # 262144,
        # 524288,
        # 1048576,
    ]
    depth_percents = np.linspace(0, 100, num=21)

    dataset_dict = {"id": [], "prompt": [], "answer": [], "context_length": [], "depth_percent": []}
    generated_ids = set()

    for context_length in context_lengths:
        print(f"Generating prompts for context length: {context_length} \n")
        for depth_percent in depth_percents:
            for _ in range(num_prompts):
                pass_key = random.randint(10000, 50000)
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
dataset.save_to_disk("needle_in_a_hay_stack_eval_dataset")
dataset.push_to_hub("nanotron/needle_in_a_hay_stack_eval_dataset")


############################################################################################################


# import glob
# import os
# import random
# import uuid

# import numpy as np
# from datasets import Dataset
# from transformers import AutoTokenizer


# if __name__ == "__main__":
#     def generate_needle_in_haystack_test(
#         needle, haystack_dir, soft_prompt: str, retrieval_question, context_length, depth_percent
#     ):
#         def read_context_files():
#             context = ""
#             base_dir = os.path.abspath(os.path.dirname(__file__))  # Package directory

#             while len(tokenizer.encode(context)) < context_length:
#                 for file in glob.glob(os.path.join(base_dir, haystack_dir, "*.txt")):
#                     with open(file, "r") as f:
#                         context += f.read()
#             return context

#         def insert_needle(context):
#             if depth_percent == 100:
#                 # If depth percent is 100, place the needle at the end
#                 new_context = context[: context_length - len(tokenizer.encode(needle))] + needle
#             else:
#                 # Get the position to insert the needle
#                 insertion_point = int(context_length * (depth_percent / 100))

#                 # Find the nearest period to the insertion point
#                 while context[insertion_point] != "." and insertion_point > 0:
#                     insertion_point -= 1

#                 # Insert the needle at the appropriate position
#                 new_context = context[:insertion_point] + needle + context[insertion_point:context_length]

#             return new_context

#         # Load up the haystack context
#         context = read_context_files()

#         # Truncate the context to the desired context length
#         context = tokenizer.decode(tokenizer.encode(context)[:context_length])

#         # Insert the needle into the context at the specified depth percent
#         context_with_needle = insert_needle(context)

#         # Generate the prompt using the context with the needle
#         prompt = f"{soft_prompt} {context_with_needle} \n\n{retrieval_question}"

#         return prompt, needle

#     # Working example
#     haystack_dir = "./"

#     # Load the tokenizer
#     tokenizer = AutoTokenizer.from_pretrained("lvwerra/the-tokenizer-v1")


#     def generate_dataset():
#         num_prompts = 1
#         soft_prompt = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
#         context_lengths = [
#             # 1024,
#             # 2048,
#             # 4096,
#             # 8192,
#             # 16384,
#             32768,
#             # 65536,
#             # 131072,
#             # 262144,
#             # 524288,
#             # 1048576,
#         ]
#         depth_percents = np.linspace(0, 100, num=21)

#         dataset_dict = {"id": [], "prompt": [], "answer": [], "context_length": [], "depth_percent": []}
#         generated_ids = set()

#         for context_length in context_lengths:
#             print(f"Generating prompts for context length: {context_length} \n")
#             for depth_percent in depth_percents:
#                 for _ in range(num_prompts):
#                     pass_key = random.randint(10000, 50000)
#                     needle = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key. "
#                     retrieval_question = f"What is the pass key? The pass key is "

#                     prompt, _ = generate_needle_in_haystack_test(
#                         needle, haystack_dir, soft_prompt, retrieval_question, context_length, depth_percent
#                     )

#                     while True:
#                         text_id = str(uuid.uuid4())
#                         if text_id not in generated_ids:
#                             generated_ids.add(text_id)
#                             break

#                     dataset_dict["id"].append(text_id)
#                     dataset_dict["prompt"].append(prompt)
#                     dataset_dict["answer"].append(pass_key)
#                     dataset_dict["context_length"].append(context_length)
#                     dataset_dict["depth_percent"].append(depth_percent)

#         dataset = Dataset.from_dict(dataset_dict)
#         return dataset


#     # Generate the dataset
#     dataset = generate_dataset()

#     # Save the dataset to disk
#     dataset.save_to_disk("needle_in_a_hay_stack_finetuning_dataset")
#     dataset.push_to_hub("nanotron/needle_in_a_hay_stack_finetuning_dataset")


############################################################################################################

# import glob
# import os
# import random
# import uuid

# import numpy as np
# from datasets import Dataset
# from transformers import AutoTokenizer

# if __name__ == "__main__":
#     def generate_needle_in_haystack_test(
#         needle, haystack_dir, soft_prompt: str, retrieval_question, context_length, depth_percent
#     ):
#         def read_context_files():
#             context = ""
#             base_dir = os.path.abspath(os.path.dirname(__file__))  # Package directory
#             files = glob.glob(os.path.join(base_dir, haystack_dir, "*.txt"))
#             file_index = 0

#             while len(tokenizer.encode(f"{soft_prompt} {context} {needle} \n\n{retrieval_question}")) < context_length:
#                 with open(files[file_index], "r") as f:
#                     context += f.read()
#                 file_index = (file_index + 1) % len(files)

#             return context

#         def insert_needle(context):
#             if depth_percent == 100:
#                 # If depth percent is 100, place the needle at the end
#                 new_context = context[: -len(needle)] + needle
#             else:
#                 # Get the position to insert the needle
#                 insertion_point = int(len(context) * (depth_percent / 100))

#                 # Find the nearest period to the insertion point
#                 while context[insertion_point] != "." and insertion_point > 0:
#                     insertion_point -= 1

#                 # Insert the needle at the appropriate position
#                 new_context = context[:insertion_point] + needle + context[insertion_point:]

#             return new_context

#         # Load up the haystack context
#         context = read_context_files()

#         # Insert the needle into the context at the specified depth percent
#         context_with_needle = insert_needle(context)

#         # Generate the prompt using the context with the needle
#         prompt = f"{soft_prompt} {context_with_needle} \n\n{retrieval_question}"

#         # Truncate the prompt to the desired context length
#         prompt = tokenizer.decode(tokenizer.encode(prompt)[:context_length])

#         return prompt, needle


#     # Working example
#     haystack_dir = "./"

#     # Load the tokenizer
#     tokenizer = AutoTokenizer.from_pretrained("lvwerra/the-tokenizer-v1")


#     def generate_dataset():
#         num_prompts = 1
#         soft_prompt = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
#         context_lengths = [
#             # 1024,
#             # 2048,
#             # 4096,
#             # 8192,
#             # 16384,
#             32768,
#             # 65536,
#             # 131072,
#             # 262144,
#             # 524288,
#             # 1048576,
#         ]
#         depth_percents = np.linspace(0, 100, num=21)

#         dataset_dict = {"id": [], "prompt": [], "answer": [], "context_length": [], "depth_percent": []}
#         generated_ids = set()

#         for context_length in context_lengths:
#             print(f"Generating prompts for context length: {context_length} \n")
#             for depth_percent in depth_percents:
#                 for _ in range(num_prompts):
#                     pass_key = random.randint(10000, 50000)
#                     needle = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key. "
#                     retrieval_question = f"What is the pass key? The pass key is "

#                     prompt, _ = generate_needle_in_haystack_test(
#                         needle, haystack_dir, soft_prompt, retrieval_question, context_length, depth_percent
#                     )

#                     while True:
#                         text_id = str(uuid.uuid4())
#                         if text_id not in generated_ids:
#                             generated_ids.add(text_id)
#                             break

#                     dataset_dict["id"].append(text_id)
#                     dataset_dict["prompt"].append(prompt)
#                     dataset_dict["answer"].append(pass_key)
#                     dataset_dict["context_length"].append(context_length)
#                     dataset_dict["depth_percent"].append(depth_percent)

#         dataset = Dataset.from_dict(dataset_dict)
#         return dataset


#     # Generate the dataset
#     dataset = generate_dataset()

#     # Save the dataset to disk
#     dataset.save_to_disk("needle_in_a_hay_stack_eval_dataset")
#     dataset.push_to_hub("nanotron/needle_in_a_hay_stack_eval_dataset")


# import glob
# import os
# import random
# import uuid

# import numpy as np
# from datasets import Dataset
# from transformers import AutoTokenizer

# if __name__ == "__main__":
#     def generate_needle_in_haystack_test(
#         needle, haystack_dir, soft_prompt: str, retrieval_question, context_length, depth_percent
#     ):
#         def read_context_files():
#             base_dir = os.path.abspath(os.path.dirname(__file__))  # Package directory
#             files = glob.glob(os.path.join(base_dir, haystack_dir, "*.txt"))
#             file_contents = []
#             for file in files:
#                 with open(file, "r") as f:
#                     file_contents.append(f.read())
#             return file_contents

#         def insert_needle(context):
#             if depth_percent == 100:
#                 # If depth percent is 100, place the needle at the end
#                 new_context = context[: -len(needle)] + needle
#             else:
#                 # Get the position to insert the needle
#                 insertion_point = int(len(context) * (depth_percent / 100))

#                 # Find the nearest period to the insertion point
#                 while context[insertion_point] != "." and insertion_point > 0:
#                     insertion_point -= 1

#                 # Insert the needle at the appropriate position
#                 new_context = context[:insertion_point] + needle + context[insertion_point:]

#             return new_context

#         # Load up the haystack context
#         file_contents = read_context_files()
#         context = "".join(file_contents)

#         # Calculate the number of tokens for soft_prompt and retrieval_question
#         soft_prompt_tokens = len(tokenizer.encode(soft_prompt))
#         retrieval_question_tokens = len(tokenizer.encode(retrieval_question))

#         # Calculate the remaining tokens for the context
#         remaining_tokens = context_length - soft_prompt_tokens - retrieval_question_tokens

#         # Repeat the context to make it long enough
#         repeated_context = (context * ((remaining_tokens // len(tokenizer.encode(context))) + 1))[:remaining_tokens]

#         # Insert the needle into the repeated context at the specified depth percent
#         context_with_needle = insert_needle(repeated_context)

#         # Generate the prompt using the context with the needle
#         prompt = f"{soft_prompt} {context_with_needle} \n\n{retrieval_question}"

#         return prompt, needle


#     # Working example
#     haystack_dir = "./"

#     # Load the tokenizer
#     tokenizer = AutoTokenizer.from_pretrained("lvwerra/the-tokenizer-v1")


#     def generate_dataset():
#         num_prompts = 1
#         soft_prompt = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
#         context_lengths = [
#             # 1024,
#             # 2048,
#             # 4096,
#             # 8192,
#             # 16384,
#             32768,
#             # 65536,
#             # 131072,
#             # 262144,
#             # 524288,
#             # 1048576,
#         ]
#         depth_percents = np.linspace(0, 100, num=21)

#         dataset_dict = {"id": [], "prompt": [], "answer": [], "context_length": [], "depth_percent": []}
#         generated_ids = set()

#         for context_length in context_lengths:
#             print(f"Generating prompts for context length: {context_length} \n")
#             for depth_percent in depth_percents:
#                 for _ in range(num_prompts):
#                     pass_key = random.randint(10000, 50000)
#                     needle = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key. "
#                     retrieval_question = f"What is the pass key? The pass key is "

#                     prompt, _ = generate_needle_in_haystack_test(
#                         needle, haystack_dir, soft_prompt, retrieval_question, context_length, depth_percent
#                     )

#                     while True:
#                         text_id = str(uuid.uuid4())
#                         if text_id not in generated_ids:
#                             generated_ids.add(text_id)
#                             break

#                     dataset_dict["id"].append(text_id)
#                     dataset_dict["prompt"].append(prompt)
#                     dataset_dict["answer"].append(pass_key)
#                     dataset_dict["context_length"].append(context_length)
#                     dataset_dict["depth_percent"].append(depth_percent)

#         dataset = Dataset.from_dict(dataset_dict)
#         return dataset


#     # Generate the dataset
#     dataset = generate_dataset()

#     # Save the dataset to disk
#     dataset.save_to_disk("needle_in_a_hay_stack_eval_dataset")
#     dataset.push_to_hub("nanotron/needle_in_a_hay_stack_eval_dataset")

import glob
import os
import random
import uuid

import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer


def token_length(tokenizer, text):
    return len(tokenizer.encode(text)[:-1])  # -1 to exclude the EOS token


def read_context_files(tokenizer, soft_prompt, retrieval_question, context_length):
    context = ""
    base_dir = os.path.abspath(os.path.dirname(__file__))
    files = glob.glob(os.path.join(base_dir, haystack_dir, "*.txt"))
    file_index = 0

    target_context_length = (
        context_length - token_length(tokenizer, soft_prompt) - token_length(tokenizer, retrieval_question)
    )

    while token_length(tokenizer, context) < target_context_length:
        with open(files[file_index], "r") as f:
            context += f.read()
        file_index = (file_index + 1) % len(files)

    return context


def insert_needle(context, needle):
    # Get the position to insert the needle
    insertion_point = random.randint(0, len(context)) - len(needle)

    # Insert the needle at the appropriate position
    new_context = context[:insertion_point] + needle + context[insertion_point:]

    return new_context


def generate_needle_in_haystack_test(
    needle, haystack_dir, soft_prompt: str, retrieval_question, context_length, depth_percent
):
    # Load up the haystack context
    tokenizer = AutoTokenizer.from_pretrained("lvwerra/the-tokenizer-v1")
    context = read_context_files(tokenizer, soft_prompt, retrieval_question, context_length)

    # Insert the needle into the context at the specified depth percent
    context_with_needle = insert_needle(context, needle)

    # Generate the prompt using the context with the needle
    prompt = f"{soft_prompt} {context_with_needle} \n\n{retrieval_question}"

    return prompt, needle


if __name__ == "__main__":
    haystack_dir = "./"

    def generate_dataset():
        num_prompts = 1
        soft_prompt = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
        context_lengths = [
            32768,
        ]
        depth_percents = np.linspace(0, 100, num=21)

        dataset_dict = {"id": [], "prompt": [], "answer": [], "context_length": [], "depth_percent": []}
        generated_ids = set()

        for context_length in context_lengths:
            print(f"Generating prompts for context length: {context_length} \n")
            for depth_percent in depth_percents:
                for _ in range(num_prompts):
                    pass_key = random.randint(10000, 50000)
                    needle = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key. "
                    retrieval_question = "What is the pass key? The pass key is "

                    prompt, _ = generate_needle_in_haystack_test(
                        needle, haystack_dir, soft_prompt, retrieval_question, context_length, depth_percent
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
    dataset.save_to_disk("needle_in_a_hay_stack_eval_dataset")
    dataset.push_to_hub("nanotron/needle_in_a_hay_stack_eval_dataset")
