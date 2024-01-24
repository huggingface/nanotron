## The internals of nanotron

### 1. Tensor Parallelism

#### Asynchronous Tensor Parallelism

Q: What are the two different tensor parallel linear modes in nanotron?

A: All-reduce and Reduce-scatter


Q: How does asynchronous column parallel linear work differently than regular column parallel linear?

A: In regular column parallel linear, each rank only computes its portion of the output matrix, then gathers the partial outputs at the end.

In asynchronous column parallel, each rank kicks off an asynchronous all-gather on the input tensor at the start. While that communication is happening, the rank computes the portion of the output corresponding to its local shard of the weights. When the all-gather finishes, each rank can compute the remaining portions of the output matrix it's missing, using the parts of the gathered input from other ranks.


Q: In asynchronous column parallel, what exactly does it gather?

A: In asynchronous column parallel, each rank kicks off an all-gather operation on the input tensor X at the start of the forward pass. This gathers the shards of X from all tensor parallel ranks into one large tensor.

For example with 4 GPUs:
+ Input X is sharded as [X0, X1, X2, X3] across 4 ranks
+ Rank 0 all-gathers: [X0, X1, X2, X3]

So each GPU gathers the complete input X from all GPUs.


Q: In nanotron, what is the core difference between regular and asynchronous tensor parallel linear layers in terms of computation?

A:
- In regular column parallel, each rank only computes the portion of the output corresponding to its shard of weights. It does not compute the full output matrix.
- In asynchronous column parallel, each rank computes the entire output matrix locally using inputs and its shard of weights.


Q: What do before_shard and after_shard represent in asynchronous tensor parallel?

A:
+ before_shard is the portion of the output matrix that a rank can compute using input shards that come before its own input shard.
+ after_shard is the portion of the output matrix that a rank can compute using input shards that come after its own input shard.

For example, on rank 2 with input shards [X0, X1, X2, X3]: before_shard = X0 * W0 + X1 * W1 after_shard = X3 * W3

Q: What is the core tradeoff between asynchronous and regular tensor parallelism?

A: Async trades off more floating point operations (FLOPs) for less communication.
It does more FLOPs by having each rank compute the full output matrix instead of just a partial shard. But it reduces communication by doing only a single collective communication. So async can improve performance if the model is communication bound, at the cost of increased FLOP requirements.


Q: Can you give a concrete example illustrating how asynchronous tensor parallelism works? (6 steps)

A:
- Step 1: Let's look at an example with 4 GPU ranks:
    + Input X sharded across ranks as [X0, X1, X2, X3]
    + Weight matrix W sharded as [W0, W1, W2, W3]
- Step 2: Rank 2 kicks off async all-gather to get [X0, X1, X2, X3]
- Step 3: While gathering, rank 2 computes: local_output = X2 * W2
- Step 4: All-gather completes, rank 2 has [X0, X1, X2, X3]
- Step 5: Rank 2 computes: before_local_output = X0 * W0 + X1 * W1, after_local_output = X3 * W3
- Step 6: Rank 2's output = before_local_output + local_output + after_local_output

So each rank computes the full output using the locally gathered X and its shard of W.

#### Tied Linear

Q: Why does brrr have only a single rank save tied linear weights instead of all ranks?

A: Tied linear weights are replicated across ranks, meaning all ranks hold the same weight values. Having every rank save the tied weight would result in the same weight being saved multiple times redundantly. So brrr designates only one rank (such as rank 0), to save the weight to avoid duplicating the same weight in the checkpoint.


Q: How does Nanotron detect tied parameters?

A: Nanotron has a base model class called NanotronModel. NanotronModel class implements a common method for accessing tied parameters (called .get_tied_parameters()) When initializing the model, Trainer calls this method to get a list of parameter names that should be tied.
For example, for a goose model, it may return ["lm_head.weight", "word_embeddings.weight"] indicating the lm head weight and word embedding weight should be tied.

Q: How does a tied linear layer differ from a regular parallel linear layer in nanotron?
A:
+ In a regular parallel linear layer, the weight matrix is sharded across ranks.
+ In a tied linear layer, the entire weight matrix is replicated on all ranks.


Q: What is the difference between a tied parameter and a regular parameter in nanotron?

A:
+ Tied parameters in nanotron are parameters that need to have their gradients synchronized (typically summed) across a specific set of ranks during training.
+ Regular parameters don't have any special synchronization requirements.


Q: When would you use tied parameters in a transformer model in nanotron?

A: Tied parameters should be used when the same weights are replicated in multiple layers of the transformer. A common example is tying the weights of the embedding layer and the final linear layer in the language modeling head.


Q: What are the different types of linear layers in nanotron and how are they different?

A: Tied linear, tensor parallel linear, and async tensor parallel linear

### 2. Pipeline Parallelism

Q: What are the four core components in brrr’s pipeline parallelism?

A:
+ PipelineBlock: Contains model computation split up over devices.
+ PipelineEngine: Orchestrate overall forward/backward passes across blocks.
+ PipelineBatchState: Stores all P2P operations
+ TensorPointer: Pointer to a tensor produced on a different device.


Q: How does PipelineEngine allow implementing different schedules like 1F1B or GPipe?

A: PipelineEngine has abstract methods like train_batch_iter and validate_batch_iter that are overridden by subclasses to implement different execution orderings.

For example, AllForwardAllBackward does all forwards first, then all backwards. 1F1B interleaves them, doing 1 forward then 1 backward. The specific scheduling logic is handled in these methods.


Q: What is the advantage of TensorPointer compared to directly sending activations after computation?

A: So TensorPointers allow pipeline stages to represent tensors produced on other ranks, and request them on-demand when needed for computation. The key benefit is lazy communication - tensors are only transferred between processes when really needed, not all upfront
The TensorPointers allow us to queue up a whole batch of communications that will happen later, instead of blocking and communicating each tensor as it is needed.


Q: How do TensorPointers interact with other components in brrr? (4 steps)

A: TensorPointer is used to represent tensors that are not locally available on the current process. It contains metadata about which process rank actually holds the real tensor data.

+ Step 1: Block A runs on rank 0, produces output tensor X
+ Step 2: Block B runs on rank 1, needs X as input
+ Step 3: In Block B's forward, X is represented as a TensorPointer pointing to rank 0. To actually get the X tensor data, Block B uses the TensorPointer to send a request to rank 0 to receive X.
+ Step 4: Rank 0 receives the request, sends X to rank 1, which populates it into Block B's input

Similarly, if Block B produces an output Y that the next Block C on rank 2 needs, it will return Y wrapped in a TensorPointer pointing to rank 1.


Q: In the forward pass, how do the four core components in brrr's pipeline parallelism work together? (5 steps)

A:
- Step 1: PipelineEngine coordinates executing the PipelineBlocks for each microbatch.
- Step 2: PipelineBlockA runs on device A, producing an activation x. It returns {"x": TensorPointer(rank=A)}
- Step 3: PipelineBlockB runs on device B. It sees the TensorPointer for x, telling it to retrieve x from device A. PipelineBlockB tells PipelineBatchState to receive x from device A.
- Step 4: PipelineEngine triggers PipelineBatchState to run communication. PipelineBatchState executes the receive operation, getting x from device A.
- Step 5: PipelineBlockB retrieves x from PipelineBatchState's buffer and continues its computation.


Q: What are the three core components of brrr's P2P communication?

A:
- P2P class: Handles sending and receiving tensors between ranks.
- TensorMetaData: Stores tensor’s metadata like shape, dtype… to interpret raw tensor data.
- Communication buffers: Reusable buffers for sending metadata and tensor data.


Q: What is the difference between PipelineBatchState and BatchTensorSendRecvState?

A: PipelineBatchState orchestrates pipeline communication across microbatches during training or inference. BatchTensorSendRecvState handles sending/receiving generic tensors in a batch.

PipelineBatchState leverages BatchTensorSendRecvState under the hood for lower-level P2P communication but adds pipeline-specific logic on top like managing activations and gradients across stages.


Q: Why does pipeline engine batch p2p communication? Isn’t at each clock cycle, there is only a single send or recv in a microbatch?

A: The pipeline engine batches P2P communication across microbatches, not within a microbatch. Within a microbatch there may be only a single send or receive between stages, but across microbatches the sends/receives can be batched.

For example, say we have a model with two pipeline stages, A and B. In microbatch 1, A sends tensor X to B. In microbatch 2, A sends tensor Y to B. Instead of sending X and Y in separate P2P operations, the pipeline engine will batch them together into one send of [X,Y].


Q: How does PipelineBlock's forward pass work? (4 steps)

A:
- Step 1: It receives inputs, which can be Tensors or TensorPointers from other ranks.
- Step 2: For any TensorPointer inputs, it uses P2P communication to fetch the actual tensor from the rank specified.
- Step 3: It runs the forward pass of the module it encapsulates, passing the tensors as inputs.
- Step 4: It returns a dict containing the outputs of the module. For ranks that didn't run this block, it returns TensorPointers instead of real tensors.


Q: How does a PipelineBlock decide to return a Tensor vs a TensorPointer? Explain

A: A PipelineBlock will return a TensorPointer if the block is running on a different pipeline rank from the one that is meant to output that tensor. Otherwise, it will return the actual Tensor
For example, say PipelineBlockA produces output X and is assigned to pipeline rank 2.
+ When running on pipeline rank 2, PipelineBlockA will return the actual Tensor X.
+ But when running on rank 1 or 3, PipelineBlockA will return a TensorPointer to rank 2 rather than the actual Tensor X data.


Q: In 3D parallelism, how does Nanotron calculate the overall loss when each microbatch has a different loss value?

A:
- Step 1: Each microbatch has its own loss value
- Step 2: The losses for each microbatch are summed together
- Step 3: The total sum is averaged across data parallelism
This represents the mean loss across all microbatches in the global batch


Q: What does PipelineBlock.rank represent?

A: PipelineBlock.rank specifies which pipeline parallel rank the block is assigned to. When initializing the model, each PipelineBlock's rank is set to place it on a particular pipeline rank.
For example, setting a block's rank to 2 means it will run on pipeline rank 2. The block's parameters will be instantiated on rank 2's device, and its forward pass will execute on rank 2.


Q: What do target_pp_ranks represent when initializing a nanotron model?

A:
target_pp_ranks specifies which subset of pipeline ranks the model should be built on. By default, the model is built on all pipeline ranks (0 to pp_size-1). But you can pass a custom list like [0, 2, 3] to build the model only on those ranks.
Concrete example: pp_size = 8, target_pp_ranks = [0, 4, 7]. This will build the model only on pipeline ranks 0, 4, and 7 out of the total 8 ranks. The intermediate ranks 1-3 and 5-6 will not have the model built on them.


#### Loading data in 3D parallelism

Q: In 3D parallelism, how does brrr sample training data for model replicas? (2 steps)

A: For example, with 2 devices, 4 microbatch size, and 100 samples:
- Step 1: It first divides the full dataset into equal chunks, one chunk per GPU.
    + Device 0 gets samples [0, 2, 4, .. 98]
    + Device 1 gets samples [1, 3, 5, .. 99]

- Step 2: Then within each GPU, samples are drawn sequentially to create micro-batches. The samples are accumulated into microbatches.
    Epoch 1:
    + Device 0 samples [0, 2, 4, 6] -> first microbatch
    + Device 1 samples [1, 3, 5, 7]

    Epoch 2:
    + Device 0 samples [8, 10, 12, 14]
    + Device 1 samples [9, 11, 13, 15]


Q: In the BRRR dataloader, why are some tensor values replaced with TensorPointers?

A: Dataloader is designed to work with BRRR's pipeline parallelism. Certain tensors like the input ids and attention mask are only needed by the first pipeline stage. Other ranks don't need the actual tensors - a TensorPointer is just a placeholder.

For example, say rank 2 is where the model input is located. Dataloader will return:
+ Rank 2: {"input_ids": <actual tensor>}
+ Other ranks: {"input_ids": TensorPointer(group_rank=2)}


Q: Given a dataset with: 100,000 samples, 10 model replicas, Micro-batch size = 16, Consumed samples so far = 10,000
How does the MegatronPretrainingSampler work concretely? (4 steps)

A:
+ Step 1: Available samples = 100,000 - 10,000 = 90,000
+ Step 2 Each model replicas gets shard of 90,000 / 10 = 9,000 samples
+ Step 3: With a microbatch size of 16, each worker samples indices 0-15, 16-31 etc. from its shard (9,000 - 18,000)…
+ Step 4: Update consumed samples after each micro-batch of 16


Q: In 3D parallelism, what's the difference between sequential and random pretraining samplers?

A: For example, with 2 GPUs, 4 microbatch size, and 8 samples:
- Sequential sampler walks through its chunk sequentially.
+ GPU 0: [0, 2, 4, 6]
+ GPU 1: [1, 3, 5, 7]

- Random sampler shuffles its chunk each epoch before sampling.
+ GPU 0: [6, 4, 0, 2] // shuffled shard
+ GPU 1: [5, 7, 1, 3]



### 3. Distributed Serialization

Q: What are the five things saved in a brrr checkpoint?

A: Model weights, optimizer state, learning rate scheduler, random number generator state, and any other misc metadata required for restoring sharded weights


Q: What are the key differences when brrr saves the weights for the 3 types of parameters?

A:
+ Regular parameters: Just directly save the full tensor normally.
+ Sharded parameters: Only save the shard owned by the first model replicas, to avoid redundancy across data parallelism.
+ Tied parameters: Only a rank in the tied group saves the weight.


Q: How does brrr reconstruct the full original unsharded tensor from the shards when loading a checkpoint?

A: When saving a sharded weight, brrr stores metadata about how the shards map to the original tensor. This includes:

Slices mapping info - Maps each shard's slice of the tensor to the corresponding slice in the original unsharded tensor. Like shard 1 covers unsharded tensor indices 0-50, etc.

During loading, BRRR uses this mapping to copy each shard into the right location in the unsharded tensor to reconstruct it.

- Step 1: Orig tensor A: [A1][A2][A3]
- Step 2: Checkpoint shards: A1 A2 A3
- Step 3: Loading:
    + A1 -> copy to indices 0-50 of A
    + A2 -> copy to indices 51-100 of A
    + A3 -> copy to indices 101-150 of A


Q: What are the three types of parameters that BRRR handles when saving checkpoints?

A: Regular parameters, sharded parameters, tied/replicated parameters


Q: How does brrr ensure all ranks start with the same initial random state for determinism? (3 steps)

A:
- Step 1: Rank 0 generates the initial state by seeding the RNG and grabbing the state tensor.
- Step 2: The state tensor is broadcast from rank 0 to all ranks.
- Step 3: Each rank loads the state tensor into its RNG.

### 4. Trainer & Model Initialization

#### Trainer

Q: What's the main idea behind brrr’s model initialization?

A: The main idea is to initialize models directly on the device and datatype we want by overriding PyTorch's default initialization. For example, by default PyTorch may initialize weights on CPU and in fp32. brrr overrides this so we can initialize directly in target precision format on GPUs from the start.


Q: How does brrr’s model initialization context manager work? (3 steps)

A:
- Step 1: Enter context: Override nn.Module register methods and tensor creation functions
- Step 2: Inside context: Modules/tensors now use overridden methods, so they initialize directly on target device/dtype
- Step 3: Exit context: Restore original nn.Module methods and tensor creation functions


Q: Which two nn.Module methods does brrr override to implement its model initialization context manager? Explain

A: brrr overrides nn.Module.register_parameter() and nn.Module.register_buffer() which are called when modules register parameters and buffers during initialization.


Q: What does kill switch do in Nanotron?

A: Kill switch is a file that the trainer periodically checks during training. If the kill switch file is detected, Trainer will:
+ Step 1: Save a checkpoint
+ Step 2: Exit training gracefully

Q: Why does brrr have the custom initialization context manager instead of just using module.to() to move models to the target device?

A: module.to() moves existing tensors to a new device. BRRR's custom initialization context manager initializes tensors directly on the target device to begin with. For example, if we want mixed precision on GPU from the start, the context manager will initialize weights in fp16 on the GPU, instead of initializing in fp32 on CPU then moving.


Q: In FP16 training, how does nanotron updates in the accumulated FP32 gradients when each parameter has an FP16 gradient? (4 steps)

A:
- Step 1: Each FP16 parameter has an associated FP32 gradient buffer allocated.
- Step 2: During backward, the FP16 gradients are accumulated into the FP32 buffer, instead of directly into the .grad attribute.
- Step 3: Before the optimizer step, nanotron copies the accumulated FP32 gradients into the .grad attribute of the FP32 copy of each parameter that will be updated.
- Step 4: The optimizer performs the update on the FP32 parameters.


#### Model Initialization


Q: In Nanotron, how does Trainer initialize a model from scratch using 3D parallelism? (5 steps)

A:
- Step 1: Create an instance of the model
- Step 2: Initialize parameters randomly (using model.init_model_randomly())
- Step 3: Mark tied parameters (using tie_parameters())
- Step 4: Sync model parameters across data parallelism with all_reduce
- Step 5: Sync tied parameters across their tied groups with all_reduce


Q: What is the high-level flow of BRRR's training loop? (3 steps) (ignore schedulers, logging…)

A:
- Step 1: Do a training step - run forward/backward pass through the model pipeline.
- Step 2: Check for kill switch file, exit if triggered.
- Step 3: Save checkpoint if current step matches interval.


Q: In 3D parallelism, how does Nanotron calculate the total number of parameters of a replicas? (2 steps)

A:
- Step 1: Sum the parameters within each pipeline stage (across tensor parallelism) ⇒ The total params for that stage.
- Step 2: Sum the parameters across pipeline stages ⇒ The total model parameters
    For example with 2 pipeline stages, 2 tensor parallel:
    + Stage 1: (TP0): 10 params, (TP1): 15 params. Sum = 25
    + Stage 2: (TP0): 20 params, (TP1): 25 params. Sum = 45
    Total params = Stage 1 + Stage 2 = (10+15) + (20+25) = 35 + 45 = 70


Q: Why does BRRR need a kill switch to terminate training? Can't we just Ctrl-C or cancel the job?

A: Kill switch provides a graceful way to terminate training without losing progress:
+ Ctrl-C stops the process immediately, risking corrupted checkpoints.
+ Cancelling the job kills all processes abruptly.
The kill switch allows: checkpoint is safely saved before terminating


Q: Why is there a second all-reduce after the first DP all-reduce during model initialization?

A: The first DP all-reduce syncs weights across data parallelism, but not within each replica. For example, it syncs embedding weights across DP ranks, but not between embeddings and lm_head within each rank. The second all-reduce specifically syncs tied weights like embeddings and lm_head within each replica.
For example, suppose we have: + [Embedding A1, LM Head A1], [Embedding A2, LM Head A2]
The first all-reduce makes
+ Embedding A1 == Embedding A2
+ LM Head A1 == LM Head A2
but not Embedding A1 == LM Head A1.The second all-reduce syncs Embedding A1 and LM Head A1, and Embedding A2 and LM Head A2.


Q: Why does BRRR issue an all-reduce across data parallelism dimension when initializing a model from scratch?

A: When initializing a model randomly, each replica (data parallel rank) can end up with different initial values due to randomness. The all-reduce (or an equivalent operation) syncs up these initial values across data parallelism, so each replica starts with the same initial weights.
For example, with 2 data parallel ranks:
+ Replica 1: Embedding weights initially [0.1, 0.3, 0.2]
+ Replica 2: Embedding weights initially [0.4, 0.1, 0.5]
After all-reduce, both will have the same initialized weights, say [0.25, 0.2, 0.35].


Q: What are the 3 pretraining samplers in brrr?

A:
- Sequential sampler: Walks through each GPU's data shard sequentially
- Random sampler: Shuffles each GPU's shard before walking through it
- Cyclic sampler: After one pass through the datasets, loops back to the beginning
