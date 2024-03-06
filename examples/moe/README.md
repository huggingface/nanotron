---
library_name: nanotron
---

# LlaMoE

Modeling code for LlaMoE to use with [Nanotron](https://github.com/huggingface/nanotron/)

## 🚀 Quickstart

```bash
# Generate a config file
python examples/moe/config_llamoe.py

# Install megablocks
pip install megablocks

# Run training
export CUDA_DEVICE_MAX_CONNECTIONS=1 # important for some distributed operations
torchrun --nproc_per_node=4 examples/moe/train_moe.py --config-file examples/moe/config_llamoe.yaml
```

## 🚀 Use your custom model
- Update the `LlaMoEConfig` class in `config_llamoe.py` to match your model's configuration
- Update the `LlaMoEForTraining` class in `modeling_llamoe.py` to match your model's architecture
- Pass the previous to the `DistributedTrainer` class in `train_moe.py`:
```python
trainer = DistributedTrainer(config_file, model_class=LlaMoEForTraining, model_config_class=LlaMoEConfig)
```
- Run training as usual


## Credits
Credits to the following repositories from which the code was adapted:
- https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py
- https://github.com/stanford-futuredata/megablocks/blob/main/megablocks/layers/dmoe.py
