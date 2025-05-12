import os

from huggingface_hub import snapshot_download

# Your model name
model_name = "Qwen/Qwen1.5-MoE-A2.7B"

os.makedirs("hf_weights", exist_ok=True)
# Target path to save the weights
save_path = "hf_weights"

# Download the model snapshot
model_path = snapshot_download(repo_id=model_name, local_dir=save_path)

print(f"Model downloaded to: {model_path}")
