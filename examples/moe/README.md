---
library_name: nanotron
---

### Benchmark

```bash
./examples/moe/benchmark_moe.sh /fsx/phuc/new_workspace/experiments/qwen_moe/benchmark/exp0a0_benhmark_num_experts_topk_and_ep_in_a_node
```

## Credits
Credits to the following repositories from which the code was adapted:
- https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py
- https://github.com/stanford-futuredata/megablocks/blob/main/megablocks/layers/dmoe.py
