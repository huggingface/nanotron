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
