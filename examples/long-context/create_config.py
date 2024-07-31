import os

from jinja2 import Environment, FileSystemLoader

# Choose the template
template_file = "llama3_template.jinja2"

# Load the template environment
current_directory = os.getcwd()
template_directory = os.path.join(current_directory, "templates")

env = Environment(loader=FileSystemLoader(template_directory))
template = env.get_template(template_file)
print("Template file: {}".format(template_file))
print()


###### First create a temple file ######
# You should define the model/tokenizer/dataset in the template file
# You can use llama3 converter to get the weights and tokenizer
tokenizer_path = "/fsx/haojun/lighteval_evaluation_model/NanotronLlama3-8B-Instruct"  # replace with your own
init_weights_path = "/fsx/haojun/lighteval_evaluation_model/NanotronLlama3-8B-Instruct"  # replace with your own
dataset_folder = "/fsx/haojun/datasets/tokenized_bytes_4B_tokens"  # replace with your own
###### end ######


############ hyper-parameter for experiments ############
experiment_name = "1M_4stages"
sequence_lengths = [65536, 131072, 524288, 1048576]  # Model sequence length
rope_thetas = [22400000.0, 80000000.0, 1000000000.0, 3600000000.0]  # base value of RoPE
train_steps = [10, 20, 30, 40]  # accumulative steps
batch_accumulation_per_replicas = [32, 32, 16, 8]  # gradient accumulation steps
micro_batch_sizes = [1, 1, 1, 1]  # batch size
sps = [4, 8, 32, 64]  # Sequence parallelism degree
tp = 8  # Tensor parallelism degree
checkpoint_intervals = [1, 1, 1, 1]

############ end ############

############ optimizer ############
lr_warmup_steps = 1
lr_decay_steps = 1
learning_rate = 0.00002
min_decay_lr = 0.00002
############ end ############


############ checkpoints/config path ############
# model weights output directory
checkpoints_path = os.path.join(current_directory, "weights", experiment_name)
checkpoints_paths = [checkpoints_path] * len(sequence_lengths)
resume_checkpoint_paths = ["null"] + [checkpoints_path] * (len(sequence_lengths) - 1)

# Config files output directory
output_dir = os.path.join(current_directory, "configs", experiment_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created config directory: {output_dir}")
############ end ############

# Ensure that we have exactly same number elements in each list to match the requirement
list_lengths = [
    len(checkpoints_paths),
    len(resume_checkpoint_paths),
    len(sequence_lengths),
    len(rope_thetas),
    len(train_steps),
    len(batch_accumulation_per_replicas),
    len(micro_batch_sizes),
    len(sps),
    len(checkpoint_intervals),
]
if not all(length == list_lengths[0] for length in list_lengths):
    raise ValueError("All input lists must have the same length.")


def format_float(value, decimal_places=5):
    return f"{value:.{decimal_places}f}"


for i in range(len(checkpoints_paths)):
    checkpoints_path = checkpoints_paths[i]
    resume_checkpoint_path = resume_checkpoint_paths[i]
    checkpoint_interval = checkpoint_intervals[i]
    batch_accumulation_per_replica = batch_accumulation_per_replicas[i]
    sequence_length = sequence_lengths[i]
    rope_theta = rope_thetas[i]
    train_step = train_steps[i]
    micro_batch_size = micro_batch_sizes[i]
    sp = sps[i]

    # Create the checkpoint path if it doesn't exist
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path, exist_ok=True)
        print(f"Created directory: {checkpoints_path}")

    variables = {
        "tokenizer_path": tokenizer_path,
        "init_weights_path": init_weights_path,
        "dataset_folder": dataset_folder,
        "checkpoints_path": checkpoints_path,
        "resume_checkpoint_path": resume_checkpoint_path,
        "checkpoint_interval": checkpoint_interval,
        "sequence_length": sequence_length,
        "rope_theta": rope_theta,
        "train_steps": train_step,
        "batch_accumulation_per_replica": batch_accumulation_per_replica,
        "micro_batch_size": micro_batch_size,
        "learning_rate": format_float(learning_rate),
        "min_decay_lr": format_float(min_decay_lr),
        "lr_warmup_steps": lr_warmup_steps,
        "lr_decay_steps": lr_decay_steps,
        "sp": sp,
        "tp": tp,
    }

    # Render the template with the provided variables
    config = template.render(variables)

    # # Define the output file name
    output_file = f"{output_dir}/config_{i}_theta={rope_theta/1e6}M_steps={train_step}_seq_len={sequence_length}.yaml"

    # Save the rendered configuration to a YAML file
    with open(output_file, "w") as f:
        f.write(config)

    print(f"Configuration file '{output_file}' has been generated.")
