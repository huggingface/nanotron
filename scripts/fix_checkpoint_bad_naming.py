"""Fixes the problem where '{type.value}_{suffix_name}.safetensors' was duplicated in checkpoint files

For example this script will change the following:
```
checkpoints/10/model/model/decoder/0/pp_block/attn/o_proj/model_model_weight.safetensors_pp-rank-0-of-1_tp-rank-0-of-2.safetensors
to
checkpoints/10/model/model/decoder/0/pp_block/attn/o_proj/model_weight_pp-rank-0-of-1_tp-rank-0-of-2.safetensors
```

Example Usage:

python scripts/fix_checkpoint_bad_naming.py /fsx/nouamane/projects/nanotron/checkpoints/10
"""

import argparse
import os
import re
from pathlib import Path


def update_checkpoint(checkpoint_dir: str):
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


def main():
    parser = argparse.ArgumentParser(description="Update checkpoint from 1.3 to 1.4")
    parser.add_argument("checkpoint_dir", type=Path, help="Path to the checkpoint directory")
    args = parser.parse_args()
    update_checkpoint(args.checkpoint_dir)


if __name__ == "__main__":
    main()
