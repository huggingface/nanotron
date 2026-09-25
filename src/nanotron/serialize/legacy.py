import os
import re
from pathlib import Path


def update_checkpoints_with_wrong_prefix(checkpoint_dir: Path):
    print(f"Updating checkpoint in {checkpoint_dir}")
    for root, _, files in os.walk(checkpoint_dir):
        for file in files:
            if file.endswith(".safetensors"):
                # r'(?<=model)_(model)' means match the string '_model' that is preceded by 'model'
                if len(re.findall(r"(?<=model)_(model)", file)) == 0:
                    continue
                # we remove second _model
                new_file = re.sub(r"(?<=model)_(model)", "", file)
                # we would have "model_weight.safetensors_pp-rank-0-of-1_tp-rank-0-of-2.safetensors"

                # let's assert we have two matches of ".safetensors"
                assert len(re.findall(r".safetensors", new_file)) == 2
                # then we remove first match
                new_file = re.sub(r".safetensors", "", new_file, count=1)
                # so that we get "model_weight_pp-rank-0-of-1_tp-rank-0-of-2.safetensors"

                print(f"Renaming {file} to {new_file}")
                os.rename(os.path.join(root, file), os.path.join(root, new_file))
