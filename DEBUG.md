        {
            "name": "run_train.py",
            "type": "python",
            "request": "launch",
            "program": "/fsx/nouamane/miniconda/envs/2-1-cu121/bin/torchrun",
            "console": "integratedTerminal",
            "justMyCode": false,
            "args": [
                "--nproc_per_node=8",
                "run_train.py",
                "--config-file=examples/config_tiny_llama.yaml",
            ],
            "env": {
                "USE_FAST": "1",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            }
        },
