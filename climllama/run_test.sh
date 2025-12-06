#!/bin/bash

# Run ClimLlama tests

# Required for TP > 1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Full model forward pass integration tests (requires GPU)
#pytest -n 0 tests/test_climllama.py::test_full_model_forward_pass[1-1-1] \
#       tests/test_climllama.py::test_full_model_forward_pass_parallel[2-1-1] \
#       tests/test_climllama.py::test_full_model_forward_pass_parallel[1-2-1] \
#       -v 2>&1 | tee test_output.log

# Context Parallelism tests (requires GPU, tests position_ids shape preservation fix)
pytest -n 0 tests/test_climllama.py::test_climllama_forward_with_context_parallelism[1-1-1-2] \
       tests/test_climllama.py::test_climllama_forward_with_context_parallelism[1-2-1-2] \
       tests/test_climllama.py::test_position_ids_shape_preserved_through_layers[1-1-1-2] \
       -v 2>&1 | tee -a test_debug_cp.log
