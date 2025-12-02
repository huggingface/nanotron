#!/bin/bash

# Run failed ClimLlama tests

pytest -n 0 tests/test_climllama.py::test_spatial_temporal_projection_dimensions[2-1-2] \
       tests/test_climllama.py::test_spatial_temporal_projection_dimensions[2-2-1] \
       tests/test_climllama.py::test_spatial_temporal_projection_dimensions[4-1-1] \
       tests/test_climllama.py::test_collator_output_structure[2-1-2] \
       tests/test_climllama.py::test_collator_output_structure[1-2-2] \
       tests/test_climllama.py::test_collator_position_ids_shape[4-1-1] \
       tests/test_climllama.py::test_collator_without_doc_masking[1-2-2] \
       tests/test_climllama.py::test_collator_label_mask_with_doc_boundaries[2-1-2] \
       tests/test_climllama.py::test_collator_non_participating_rank[1-2-2] \
       tests/test_climllama.py::test_collator_non_participating_rank[2-1-2] \
       tests/test_climllama.py::test_collator_position_ids_shape[1-2-2] \
       tests/test_climllama.py::test_collator_label_mask_with_doc_boundaries[1-2-2] \
       tests/test_climllama.py::test_collator_position_ids_shape[2-2-1] \
       tests/test_climllama.py::test_collator_position_ids_shape[2-1-2] \
       tests/test_climllama.py::test_collator_without_doc_masking[2-1-2] \
       -v 2>&1 | tee test_output.log
