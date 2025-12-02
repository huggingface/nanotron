#!/bin/bash

# Run ClimLlama tests

# Full model forward pass integration tests (requires GPU)
pytest -n 0 tests/test_climllama.py::test_full_model_forward_pass[1-1-1] \
       tests/test_climllama.py::test_full_model_forward_pass_parallel[2-1-1] \
       tests/test_climllama.py::test_full_model_forward_pass_parallel[1-2-1] \
       tests/test_climllama.py::test_hybrid_pe_absolute_plus_rope[1-1-1] \
       -v 2>&1 | tee test_output.log
