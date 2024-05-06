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
        <a href="#Development-guidelines">Contributions</a> •
        <a href="docs/debugging.md">Debugging</a>
    <p>
</h4>

<h3 align="center">
    <a href="https://huggingface.co/nanotron"><img style="float: middle; padding: 10px 10px 10px 10px;" width="60" height="55" src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.png" /></a>
</h3>
<h3 align="center">
<p>Pretraining models made easy
</h3>


Nanotron is a library for pretraining transformer models. It provides a simple and flexible API to pretrain models on custom datasets. Nanotron is designed to be easy to use, fast, and scalable. It is built with the following principles in mind:

- **Simplicity**: Nanotron is designed to be easy to use. It provides a simple and flexible API to pretrain models on custom datasets.
- **Performance**: Optimized for speed and scalability, Nanotron uses the latest techniques to train models faster and more efficiently.

## Installation

```bash
# Requirements: Python>=3.10
git clone https://github.com/huggingface/nanotron
cd nanotron
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121
pip install -e .

# Install dependencies if you want to use the example scripts
pip install datasets transformers
pip install "flash-attn>=2.5.0" --no-build-isolation
```
> [!NOTE]
> If you get `undefined symbol: ncclCommRegister` error you should install torch 2.1.2 instead: `pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121`

> [!TIP]
> We log to wandb automatically if it's installed. For that you can use `pip install wandb`. If you don't want to use wandb, you can run `wandb disabled`.

## Quick Start
### Training a tiny Llama model
The following command will train a tiny Llama model on a single node with 8 GPUs. The model will be saved in the `checkpoints` directory as specified in the config file.
```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=8 run_train.py --config-file examples/config_tiny_llama.yaml
```

### Run generation from your checkpoint
```bash
torchrun --nproc_per_node=1 run_generate.py --ckpt-path checkpoints/10/ --pp 1 --tp 1
```
> [!TIP]
> We could set a larger TP for faster generation, and a larger PP in case of very large models.

## Config file description
