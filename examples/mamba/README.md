---
library_name: nanotron
---

# Mamba

Modeling code for Mamba to use with [Nanotron](https://github.com/huggingface/nanotron/)

## 🚀 Quickstart

```bash
# Generate a config file
python examples/moe/config_mamba.py

pip install einops
pip install causal-conv1d>=1.1.0,<1.2.0
pip install mamba-ssm

# Run training
./examples/mamba/train_mamba.sh
```

## Credits
Credits to the following repositories from which the code was adapted:
- https://github.com/state-spaces/mamba
