## Pipeline parallelism

We choose to mimic the "torch" eager semantics:
 - Declare module/blocks at init time
 - Declare edges during forward

# Scheduling

# All forward, all backward (easy, but memory expensive)

# 1f1b (much nicer)

We're going to assume that all Pipeline blocks are assigned to a rank in a contiguous manner.

Warmup:
```
Rank 1: [forward(), forward(), forward(), forward(), backward()]
Rank 2: [forward(), forward(), forward(), forward(), backward(), backward()]
Rank 3: [forward(), forward(), forward(), forward(), backward(), backward(), backward()]
Rank 4: [forward(), backward(), forward(), backward(), forward(), backward()]
```
// TODO @thomasw21: How do we extrapolate this notion to a tree. Not sure exactly, but topological ordering should be fine

# TODOs:

- [ ] passing activation that don't require backward screws me as 1f1b works because you have the same number of forward and the same number of backward (in the stage sense)
