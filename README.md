# nanotron

The objective of this repository is to provide easy distributed primitives in order to train a variety of models efficiently.

# Philosophy

- Make it fast. At least as fast as other open source versions.
- Make it minimal. We don't actually need to support all techniques and all versions of 3D parallelism. What matters is that we can efficiently use the "best" ones.
- Make everything explicit instead of transparent. As we move forward, making things transparent works well when it works well but is a horrible debugging experience if one doesn't understand the implications of techniques used. In order to mitigate this, we choose to be explicit in the way it does things

# Core Features

We support the following:
 - 3D parallelism, including one-forward-one-backward pipeline engine
 - ZeRO-1 optimizer
 - FP32 gradient accumulation
 - Parameter tying/sharding

# Examples

<!-- ls examples/
config_nouamane_llama_tflops.yaml  config_tiny_llama.py  config_tiny_llama.yaml  train_tiny_llama.sh -->
In the `/examples` directory, you can find an example configuration file, and a script to run it. You can run it using `torchrun`:
```bash
torchrun --nproc_per_node=8 run_train.py --config-file examples/config_tiny_llama.yaml
torchrun --nproc_per_node=8 run_train.py --config-file examples/config_tiny_llama.py
```
<!-- here I should explain that user can use USE_FAST=1 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=8 run_train.py --config-file examples/config_nouamane_llama_tflops.yaml -->
*

> Note: Most examples include a slow modeling (No dependencies, only Pytorch), and a fast modeling (Flash Attention, ...). Make sure to install the dependencies if you want to run the fast modeling, then set the env `export USE_FAST=1`

# Installation

Requirements:
 - Python >= 3.10
 - PyTorch >= 2.0.0
 - Flash-Attention >= 2.4.2

To install:
```bash
git clone git@github.com:huggingface/nanotron.git
cd nanotron
pip install -e .
```

Install also:
- Flash Attention: `pip install packaging; pip install flash-attn>=2.4.2  --no-build-isolation`
- Also good to have `transformers` `datasets` `python-etcd` `tensorboardX`: `pip install transformers datasets python-etcd tensorboardX`


We also support a set of flavors that you can install using `pip install -e [$FLAVOR]`:
 - `dev`: Used is you are developping in `nanotron`. It installs in particular our linter mechanism. On top of that you have to run `pre-commit install` afterwards.
 - `test`: We use `pytest` in order to run out testing suite. In order to run tests in parallel, it will install `pytest-xdist`, which you can leverage by running `pytest -n 12 tests` (12 is the number of parallel test)


# Development guidelines

If you plan on developping on `nanotron`, we suggest you install the `dev` flavor: `pip install -e ".[dev]"`

We use pre-commit to run a bunch of callbacks on each commit, mostly normalization code in order for the codebase to stay consistent. Please do run `pre-commit install`.

For the linting:
```bash
pre-commit install
pre-commit run --config .pre-commit-config.yaml --all-files
```

# Credits

We would like to thank everyone working on LLMs, especially those sharing their work openly from which we took great inspiration: Nvidia for `Megatron-LM/apex`, Microsoft for `DeepSpeed`, HazyResearch for `flash-attn`
