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
from pathlib import Path

from nanotron.serialize.legacy import update_checkpoints_with_wrong_prefix as update_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Update checkpoint from 1.3 to 1.4")
    parser.add_argument("checkpoint_dir", type=Path, help="Path to the checkpoint directory")
    args = parser.parse_args()
    update_checkpoint(args.checkpoint_dir)


if __name__ == "__main__":
    main()
