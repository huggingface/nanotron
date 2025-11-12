# Implementation Plan: Node-Level File Separation for IndexedDataset

## Current Architecture Summary

Currently:
- **All ranks access all files** via `BlendableDataset`
- Distribution happens at the **sample level** through `DistributedSampler`
- Each node/rank must have access to all `.bin` files

## Proposed Architecture

**Goal:** Partition files across nodes so that:
1. Each node gets a **disjoint set** of files
2. Files are assigned to nodes based on **accumulated weights** ≥ `total_weights / num_nodes`
3. Within a node, ranks use `DistributedSampler` to distribute samples randomly

---

## Implementation Plan

### **Step 1: Design Node-Level File Partitioning Strategy**

Create a new utility: `src/nanotron/data/node_partitioner.py`

**Key functions:**
```python
def partition_files_by_node(
    data_prefix: List,  # [weight1, path1, weight2, path2, ...]
    num_nodes: int,
    node_rank: int,
) -> List:
    """
    Partition files across nodes by accumulated weights.

    Algorithm:
    1. Parse weighted data_prefix
    2. Sort files by weight (descending) for better load balancing
    3. Use greedy bin packing: assign each file to node with smallest accumulated weight
    4. Ensure each node gets at least total_weights/num_nodes
    5. Return subset of data_prefix for current node_rank

    Returns: [weight1, path1, weight2, path2, ...] for this node only
    """
```

**Example:**
```python
Input:
  data_prefix = [0.5, "file1", 0.3, "file2", 0.15, "file3", 0.05, "file4"]
  num_nodes = 2

Node 0 gets: [0.5, "file1", 0.05, "file4"]  # total weight = 0.55
Node 1 gets: [0.3, "file2", 0.15, "file3"]  # total weight = 0.45
```

---

### **Step 2: Calculate Node Rank from ParallelContext**

Add utility function in `src/nanotron/data/node_partitioner.py`:

```python
def get_node_info(parallel_context: ParallelContext) -> tuple[int, int]:
    """
    Calculate node_rank and num_nodes from parallel context.

    Assumptions:
    - local_world_size = GPUs per node
    - world_size = total GPUs
    - num_nodes = world_size / local_world_size
    - node_rank = global_rank // local_world_size

    Returns: (node_rank, num_nodes)
    """
    local_world_size = parallel_context.local_world_size
    world_size = parallel_context.world_size
    global_rank = dist.get_rank()

    num_nodes = world_size // local_world_size
    node_rank = global_rank // local_world_size

    return node_rank, num_nodes
```

---

### **Step 3: Modify `build_dataset()` to Support Node Partitioning**

Update `src/nanotron/data/nemo_dataset/__init__.py`:

```python
def build_dataset(
    cfg: Any,
    tokenizer: PreTrainedTokenizerBase,
    data_prefix: List[str],
    num_samples: int,
    seq_length: int,
    seed: Any,
    skip_warmup: bool,
    name: str,
    parallel_context: ParallelContext,
    enable_node_partitioning: bool = False,  # NEW PARAMETER
) -> Union["GPTDataset", BlendableDataset]:

    # NEW: Apply node-level partitioning if enabled
    if enable_node_partitioning and len(data_prefix) > 1:
        from nanotron.data.node_partitioner import partition_files_by_node, get_node_info

        node_rank, num_nodes = get_node_info(parallel_context)

        log_rank(
            f"Node partitioning enabled: {num_nodes} nodes, current node_rank={node_rank}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        # Partition files for this node only
        data_prefix = partition_files_by_node(
            data_prefix=data_prefix,
            num_nodes=num_nodes,
            node_rank=node_rank,
        )

        log_rank(
            f"Node {node_rank} assigned {len(data_prefix)//2} files",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

    # Rest of the function remains the same
    # Each node now builds BlendableDataset with its own subset of files
    ...
```

---

### **Step 4: Add Configuration Option**

Update `src/nanotron/config/config.py` to add `IndexedDatasetsArgs`:

```python
@dataclass
class IndexedDatasetsArgs:
    data_prefix: Union[str, List[str]]
    splits_string: str = "969,30,1"
    validation_drop_last: bool = True
    eod_mask_loss: bool = False
    no_seqlen_plus_one_input_tokens: bool = False
    index_mapping_dir: Optional[str] = None
    skip_warmup: bool = False
    enable_node_partitioning: bool = False  # NEW FIELD
```

---

### **Step 5: Update `prepare_training_config.py`**

Add CLI argument and pass through to config:

```python
parser.add_argument(
    "--enable_node_partitioning",
    action="store_true",
    help="Enable node-level file partitioning (each node gets disjoint files)",
)

# In create_training_config():
data=DataArgs(
    dataset=IndexedDatasetsArgs(
        data_prefix=parsed_data_prefix,
        splits_string=splits_string,
        validation_drop_last=True,
        eod_mask_loss=False,
        no_seqlen_plus_one_input_tokens=False,
        index_mapping_dir=index_mapping_dir,
        skip_warmup=skip_warmup,
        enable_node_partitioning=enable_node_partitioning,  # NEW
    ),
    seed=seed,
),
```

---

### **Step 6: Update `run_train.py` to Pass Parameter**

Modify `run_train.py:189-207`:

```python
train_dataset = build_dataset(
    cfg=data.dataset,
    tokenizer=tokenizer,
    data_prefix=data.dataset.data_prefix,
    num_samples=trainer.config.tokens.train_steps * trainer.global_batch_size,
    seq_length=trainer.sequence_length,
    seed=data.seed,
    skip_warmup=data.dataset.skip_warmup,
    name="train",
    parallel_context=trainer.parallel_context,
    enable_node_partitioning=data.dataset.enable_node_partitioning,  # NEW
)
```

---

## Key Design Decisions

### **1. Greedy Bin Packing Algorithm**

For load balancing, we'll use a **greedy bin packing** approach:

```python
def partition_files_by_node(data_prefix: List, num_nodes: int, node_rank: int) -> List:
    # Parse weights and paths
    files = []  # [(weight, path), ...]
    for i in range(0, len(data_prefix), 2):
        weight = float(data_prefix[i])
        path = data_prefix[i + 1]
        files.append((weight, path))

    # Sort by weight descending for better packing
    files.sort(key=lambda x: x[0], reverse=True)

    # Initialize node bins
    node_bins = [[] for _ in range(num_nodes)]
    node_weights = [0.0 for _ in range(num_nodes)]

    # Greedy assignment: assign each file to node with smallest weight
    for weight, path in files:
        min_node = node_weights.index(min(node_weights))
        node_bins[min_node].append((weight, path))
        node_weights[min_node] += weight

    # Return files for requested node_rank
    result = []
    for weight, path in node_bins[node_rank]:
        result.extend([weight, path])

    return result
```

### **2. Handling Edge Cases**

- **Single file:** No partitioning needed, all nodes get same file
- **Fewer files than nodes:** Some nodes may get no files (need to handle gracefully)
- **Unequal weights:** Greedy algorithm ensures relatively balanced distribution

### **3. Sample Distribution Within Node**

After node-level partitioning:
- All ranks **within a node** still access all node's files
- `DistributedSampler` distributes samples across DP ranks within node
- No change to existing sample-level distribution logic

---

## Benefits

✅ **Reduced storage requirements:** Each node only needs its assigned files
✅ **Reduced I/O contention:** Nodes read different files
✅ **Maintains load balancing:** Greedy algorithm ensures weights are balanced
✅ **Backward compatible:** Disabled by default with `enable_node_partitioning=False`
✅ **Simple integration:** Minimal changes to existing codebase

---

## Example Usage

```bash
python climllama/prepare_training_config.py \
    --checkpoint_path /path/to/checkpoint \
    --data_prefix "0.5,/data/file1,0.3,/data/file2,0.15,/data/file3,0.05,/data/file4" \
    --enable_node_partitioning \
    --output_config config_train.yaml
```

**Result with 2 nodes:**
- Node 0: Loads `file1` (weight=0.5) + `file4` (weight=0.05) = 0.55
- Node 1: Loads `file2` (weight=0.3) + `file3` (weight=0.15) = 0.45

---

## Testing Strategy

1. **Unit tests** for `partition_files_by_node()`:
   - Test with 2, 4, 8 nodes
   - Test with unequal weights
   - Test edge cases (single file, more nodes than files)

2. **Integration test** with small dummy `.bin` files:
   - Create 4 dummy IndexedDataset files with different sizes
   - Run on 2 nodes
   - Verify each node only loads its assigned files
   - Verify training completes successfully

3. **Validation checks:**
   - Sum of weights per node ≥ `total_weight / num_nodes`
   - Files are disjoint across nodes
   - All files are assigned (none dropped)

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/nanotron/data/node_partitioner.py` | **CREATE** | New utility for node-level file partitioning |
| `src/nanotron/data/nemo_dataset/__init__.py` | **MODIFY** | Add `enable_node_partitioning` parameter to `build_dataset()` |
| `src/nanotron/config/config.py` | **MODIFY** | Add `enable_node_partitioning` field to `IndexedDatasetsArgs` |
| `climllama/prepare_training_config.py` | **MODIFY** | Add CLI arg and pass to config |
| `run_train.py` | **MODIFY** | Pass parameter from config to `build_dataset()` |
| `tests/test_node_partitioner.py` | **CREATE** | Unit tests for partitioning logic |

---

## Implementation Order

1. Create `node_partitioner.py` with core algorithms
2. Add config field to `IndexedDatasetsArgs`
3. Modify `build_dataset()` to use partitioner
4. Update `prepare_training_config.py` CLI
5. Update `run_train.py` to pass parameter
6. Write unit tests
7. Integration testing

---

## Architecture Diagram

```
Before (Current):
┌─────────────────────────────────────┐
│ All Nodes                            │
│  ├─ BlendableDataset                │
│  │   ├─ GPTDataset(file1.bin) [70%]│
│  │   ├─ GPTDataset(file2.bin) [20%]│
│  │   └─ GPTDataset(file3.bin) [10%]│
│  └─ DistributedSampler              │
│      ├─ Rank 0: samples [0,4,8,...]│
│      ├─ Rank 1: samples [1,5,9,...]│
│      └─ Rank 2: samples [2,6,10,..]│
└─────────────────────────────────────┘
Each rank may access any file

After (Node Partitioning):
┌─────────────────────────────────────┐
│ Node 0                               │
│  ├─ BlendableDataset                │
│  │   ├─ GPTDataset(file1.bin) [70%]│
│  │   └─ GPTDataset(file3.bin) [10%]│
│  └─ DistributedSampler              │
│      ├─ Rank 0: samples [0,2,4,...] │
│      └─ Rank 1: samples [1,3,5,...] │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ Node 1                               │
│  ├─ BlendableDataset                │
│  │   └─ GPTDataset(file2.bin) [20%]│
│  └─ DistributedSampler              │
│      ├─ Rank 2: samples [0,2,4,...] │
│      └─ Rank 3: samples [1,3,5,...] │
└─────────────────────────────────────┘

Each node accesses disjoint files
```

---

## Additional Considerations

### **Checkpointing and Resumption**

When resuming training:
- File partitioning should be **deterministic** (same seed → same partition)
- Use node topology info (num_nodes, node_rank) consistently
- Document that changing num_nodes requires restarting from scratch

### **Validation Data**

Apply same partitioning logic to validation datasets to ensure consistency.

### **Logging and Debugging**

Add detailed logging:
```python
log_rank(
    f"Node {node_rank}/{num_nodes} assigned files:\n" +
    "\n".join([f"  - {weight:.3f}: {path}" for weight, path in node_files]),
    logger=logger,
    level=logging.INFO,
    rank=0,
)
```

### **Error Handling**

- Raise clear errors if `num_nodes > num_files`
- Warn if weight distribution is highly imbalanced (>20% deviation)
- Validate that all file paths exist before partitioning

---

## Performance Expectations

**Storage Savings:**
- With N nodes: Each node stores ~1/N of total data
- Example: 4 nodes, 1TB total → 250GB per node

**I/O Benefits:**
- Reduced network/filesystem contention
- Each node reads independent files
- Better cache locality per node

**Training Speed:**
- Similar to current approach (DistributedSampler unchanged within node)
- Potential slight improvement from reduced I/O contention
