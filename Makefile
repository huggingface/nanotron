# Run nanotron's tests and examples's tests
test:
	pytest \
        --color=yes \
        --durations=0 \
        --ignore tests/fp8 \
        --verbose \
        tests/

	pip install -r examples/doremi/requirements.txt
	pytest \
        --color=yes \
        --durations=0 \
        --ignore tests/fp8 \
        --verbose \
        examples/doremi/tests/

	pip install -r examples/llama/requirements.txt
	pytest \
        --color=yes \
        --verbose \
        examples/llama/tests/

install-moe:
	pip install --no-build-isolation git+https://github.com/fanshiqing/grouped_gemm@main

test-moe:
	pytest --color=yes --verbose tests/test_moe_dispatcher.py
	pytest --color=yes --verbose tests/test_moe.py
	pytest --color=yes --verbose tests/test_distributed_primitives.py::test_all_to_all

run-sanity-moe:
	CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=8 run_train.py --config-file /fsx/phuc/new_workspace/snippets/experiment_configs/qwen_moe/exp0a0_sanity_dense.yaml
	CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=8 run_train.py --config-file /fsx/phuc/new_workspace/snippets/experiment_configs/qwen_moe/exp0b0_sanity_moe_ep8.yaml
