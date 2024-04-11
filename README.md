<h1 align="center">⚡️ Nanotron</h1>

<p align="center">
    <a href="https://github.com/huggingface/nanotron/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/nanotron.svg">
    </a>
    <a href="https://github.com/huggingface/nanotron/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/huggingface/nanotron.svg?color=green">
    </a>
</p>

<h4 align="center">
    <p>
        <a href="#Philosophy">Philosophy</a> •
        <a href="#Core-Features">Core Features</a> •
        <a href="#Installation">Installation</a> •
        <a href="#Quick-examples">Usage</a> •
        <a href="#Development-guidelines">Contributions</a>
    <p>
</h4>

<h3 align="center">
    <a href="https://huggingface.co/nanotron"><img style="float: middle; padding: 10px 10px 10px 10px;" width="60" height="55" src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.png" /></a>
</h3>



#

The objective of this library is to provide easy distributed primitives in order to train a variety of models efficiently using 3D parallelism. For more information about the internal design of the library or 3D parallelism in general, please check out [[docs.md]](./docs/docs.md) and [[3d_parallelism.md]](./docs/3d_parallelism.md).


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

# Installation

Requirements:
 - Python >= 3.10
 - PyTorch >= 2.0.0
 - Flash-Attention >= 2.5.0

To install (in a new env):
```bash
pip install torch
pip install packaging; pip install "flash-attn>=2.5.0"  --no-build-isolation
pip install nanotron
```

Also nice to have: `pip install transformers datasets python-etcd tensorboardX`

We also support a set of flavors that you can install using `pip install -e [$FLAVOR]`:
 - `dev`: Used is you are developping in `nanotron`. It installs in particular our linter mechanism. On top of that you have to run `pre-commit install` afterwards.
 - `test`: We use `pytest` in order to run out testing suite. In order to run tests in parallel, it will install `pytest-xdist`, which you can leverage by running `pytest -n 12 tests` (12 is the number of parallel test)


# Quick examples

In the `/examples` directory, you can find a few example configuration file, and a script to run it.

You can run a sample training using:
```bash
torchrun --nproc_per_node=8 run_train.py --config-file examples/debug_run_train.yaml
```

And run a sample generation using:
```bash
torchrun --nproc_per_node=8 run_generation.py --ckpt-path checkpoints/text/4
```

# Development guidelines

If you plan on developing on `nanotron`, we suggest you install the `dev` flavor: `pip install -e ".[dev]"`

We use pre-commit to run a bunch of callbacks on each commit, mostly normalization code in order for the codebase to stay consistent. Please do run `pre-commit install`.

For the linting:
```bash
pre-commit install
pre-commit run --config .pre-commit-config.yaml --all-files
```

*As a part of making sure we aren't slowed down as the codebase grows, we will not merge a PR if the features it introduces do not have test coverage.*

We have extensions built on top of Nanotron, with their tests located in the `/examples` folder. Since VSCode defaults to discovering tests only in the `/tests` folder, please run tests from both `/examples` and `/tests` to ensure your PR does not break these extensions. Please run `make tests` to execute all the nanotron tests and the tests in the `/examples` directory that you need to pass.

Features we would like to add:
- [ ] Support `torch.compile`
- [ ] More optimized kernels
- [ ] Support Zero3
- [ ] Other PP schedules (such as Interleaved 1f1b...)
- [ ] Ring attention / Sequence Parallelism
- [ ] 3D Parallel MoEs
- [ ] Supporting more architectures (Mamba..)
- [ ] ...


# Useful scripts
- `scripts/log_lighteval_to_wandb.py`: logs the evaluation results of LightEval to wandb, including summary statistics.


# Credits

We would like to thank everyone working on LLMs, especially those sharing their work openly from which we took great inspiration: Nvidia for `Megatron-LM/apex`, Microsoft for `DeepSpeed`, HazyResearch for `flash-attn`
