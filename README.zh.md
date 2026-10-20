<h1 align="center">⚡️ Nanotron</h1>

<p align="center">
    <a href="https://github.com/huggingface/nanotron/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/nanotron.svg">
    </a>
    <a href="https://github.com/huggingface/nanotron/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/huggingface/nanotron.svg?color=green">
    </a>
</p>

<p align="center">
    <a href="README.md">English</a> • <a href="README.zh.md">简体中文</a>
</p>

<h4 align="center">
    <p>
        <a href="#安装">安装</a> •
        <a href="#快速入门">快速入门</a> •
        <a href="#特性">特性</a> •
        <a href="#基准测试">基准测试</a> •
        <a href="CONTRIBUTING.md">贡献指南</a>
    <p>
</h4>

<h3 align="center">
    <a href="https://huggingface.co/nanotron"><img style="float: middle; padding: 10px 10px 10px 10px;" width="60" height="55" src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.png" /></a>
</h3>
<h3 align="center">
<p>让模型预训练化繁为简
</h3>

Nanotron 是一个用于 Transformer 模型预训练的轻量高效库。它提供了简单灵活的 API，用于在自定义数据集上预训练模型。Nanotron 旨在兼顾易用性、高速度与可扩展性，其设计遵循以下原则：

- **简单易用（Simplicity）**：Nanotron 具有极高的易用性，提供了清晰直观的 API，便于在自定义数据集上轻松预训练模型。
- **极致性能（Performance）**：针对速度和扩展性进行深度优化，Nanotron 采用前沿技术，使模型训练更加快速高效。

📚 **欢迎查阅我们的 [Ultrascale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)** —— 使用 Nanotron 高效扩展 LLM 训练的权威指南！

📝 **由 [DeepWiki](https://deepwiki.com/huggingface/nanotron) 生成的 AI 文档**

## 安装

要运行本项目代码，首先建议使用如 `uv` 创建 Python 虚拟环境：

```shell
uv venv nanotron --python 3.11 && source nanotron/bin/activate && uv pip install --upgrade pip
```

> [!TIP]
> 如果您使用的是 Hugging Face 集群，建议在 `.bashrc` 中添加 `export UV_LINK_MODE=copy`，以消除 `uv` 的缓存提示警告。

接下来安装 PyTorch：

```shell
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

然后安装核心依赖：

```shell
uv pip install -e .
```

如需运行示例脚本，请按照以下命令安装其余依赖：

```shell
uv pip install datasets transformers datatrove[io] numba wandb
# 融合算子 (Fused kernels)
uv pip install ninja triton "flash-attn>=2.5.0" --no-build-isolation
```

接下来，登录您的 Hugging Face 与 Weights and Biases 账户：

```shell
huggingface-cli login
wandb login
```

最后，检查系统是否已安装 Git LFS，以便能够向 Hugging Face Hub 加载和推送模型及数据集：

```shell
git-lfs --version
```

如果未安装，请执行：

```shell
sudo apt-get install git-lfs
```

## 快速入门

### 训练微型 Llama 模型 (Tiny Llama)

以下命令将在单台配备 8 张 H100 GPU 的节点上训练一个微型 Llama 模型，耗时约 10 分钟：

```shell
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=8 run_train.py --config-file examples/config_tiny_llama.yaml
```

模型将保存在配置文件中指定的 `checkpoints` 目录中。

> [!NOTE]
> 您可以使用 `examples/config_tiny_llama.py` 来生成属于您自己的训练配置文件。

有关训练首个模型的详细说明，请参阅我们的 [首个模型训练指南 (Your First Training guide)](docs/your-first-training.md)。如需使用 Slurm 进行多节点训练，请参阅 [多节点训练指南 (Multi-Node Training guide)](docs/multi-node-training.md)。

### 从检查点运行文本生成

```shell
torchrun --nproc_per_node=1 run_generate.py --ckpt-path checkpoints/{checkpoint_number}/ --tp 1 --pp 1
```

增大 `--tp`（张量并行，Tensor Parallel）的值可以通过多卡加速生成，而对于超大模型，可以使用更大的 `--pp`（流水线并行，Pipeline Parallel）值。

### 使用 VSCode 进行调试

如需在 VSCode 中进行调试，请将以下配置添加到您的 `launch.json` 文件中：

```json
{
    "name": "run_train.py",
    "type": "python",
    "request": "launch",
    "program": "torchrun", // 或运行 `which torchrun` 获取 torchrun 的绝对路径
    "console": "integratedTerminal",
    "justMyCode": false,
    "args": [
        "--nproc_per_node=2",
        "run_train.py",
        "--config-file=examples/config_tiny_llama.yaml", // 或使用 examples/config_tiny_llama.py 生成自定义配置
    ],
    "env": {
        // "NANOTRON_BENCHMARK": "1", // 开启后可对训练进行数步基准评测
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "WANDB_MODE": "disabled",
    }
},
```

> [!NOTE]
> 更多信息请参阅 [Nanotron 调试示例（多 GPU）](/examples/contributor-guide/README.md#debugging-nanotron-example-on-multiple-gpus)。

### 自定义示例

您可以在 [`/examples`](/examples) 目录中找到更多示例：

| 示例 | 描述 |
| --- | --- |
| `custom-dataloader` | 为 Nanotron 接入自定义数据加载器 |
| `datatrove` | 使用 datatrove 库加载数据 |
| `doremi` | 使用 DoReMi 算法加速模型训练 |
| `mamba` | 训练 Mamba 架构模型示例 |
| `moe` | 训练混合专家（MoE）架构模型示例 |
| `mup` | 使用谱 µTransfer (spectral µTransfer) 扩展模型规模 |
| `examples/config_tiny_llama_with_s3_upload.yaml` | 自动上传检查点至 S3 |

我们正在积极添加更多示例！欢迎提交 PR 分享您的示例。🚀

## 基准测试

我们针对各种模型规模和配置对 Nanotron 进行了全面的基准评测。完整的基准测试数据、配置与日志均可在我们的 [ultrascale-playbook-data](https://huggingface.co/datasets/nanotron/ultrascale-playbook-data/tree/main) 仓库中获取。

![Model Efficiency Benchmarks](docs/benchmark_summary.svg)

上图展示了我们在 Nanotron v0.5 中针对各个模型规模和节点数量探索出的最优配置，突出了最佳 MFU（Model FLOPS Utilization，模型算力利用率）和显存占用。这些代表了通过全面基准测试所确定的最高效训练方案。敬请期待即将推出的更多性能优化！🚀

关于从这些基准测试中总结出的详细分析与最佳实践，请参阅我们的 [Ultrascale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)。

## 特性

目前已支持以下特性：
- [x] 3D 并行（DP 数据并行 + TP 张量并行 + PP 流水线并行）
- [x] 面向 MoE 的专家并行（Expert Parallelism）
- [x] 面向流水线并行的 AFAB 与 1F1B 调度算法
- [x] 显式 TP 与 PP API，便于轻松调试
- [x] ZeRO-1 优化器
- [x] FP32 梯度累积
- [x] 参数绑定 / 分片（Parameter tying / sharding）
- [x] 面向大模型的自定义模块检查点保存（Checkpointing）
- [x] 用于大模型扩展的谱 µTransfer（Spectral µTransfer）参数化方法
- [x] Mamba 示例
- [x] 基于 CUDA Event 的精准计时，实现精确的 GPU 性能评估

路线图计划支持：
- [ ] FP8 训练
- [ ] ZeRO-3 优化器（即 FSDP）
- [ ] `torch.compile` 支持
- [ ] 环形注意力（Ring Attention）
- [ ] 交错式 1F1B 调度算法（Interleaved 1F1B schedule）

## 致谢

谨向所有致力于大语言模型（LLM）研发的同仁致谢，特别是向那些无私分享研究成果、为我们带来巨大启发的开源项目致敬：Nvidia 的 `Megatron-LM/apex`、Microsoft 的 `DeepSpeed` 以及 HazyResearch 的 `flash-attn` 等。

---

> **免责声明 / Disclaimer**：本项目由开源社区志愿者以非官方身份发起并独立维护。若翻译内容与英文原版存在任何出入，请以英文官方文档为准。欢迎提交 PR 协同完善！
