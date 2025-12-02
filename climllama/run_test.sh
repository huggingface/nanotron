#!/bin/bash

# Run ClimLlama tests

# Required for TP > 1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Full model forward pass integration tests (requires GPU)
pytest -n 0 tests/test_climllama.py::test_full_model_forward_pass[1-1-1] \
       tests/test_climllama.py::test_full_model_forward_pass_parallel[2-1-1] \
       tests/test_climllama.py::test_full_model_forward_pass_parallel[1-2-1] \
       -v 2>&1 | tee test_output.log
