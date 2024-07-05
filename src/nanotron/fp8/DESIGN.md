
- For parameters like input embedding, where we are just indexing to get the corresponding embedding vectors, FP8 doesn't have to speed up the matmul since this is not a GEMM operation.
- We only keep master weights of FP8 modules. For non-FP8 modules, we directly keep them in float16.

### Key Technical Details

- Selectively choose a suitable FP8 format for weights, gradients, and activations.
- Selectively choose which layers should be in FP8.
- Perform delayed and dynamic quantization on the fly.
- Use mixed precision training for FP8 parameters (we don't keep a master weight for non-FP8 parameters).
- Loss scaling.
- Direct communication in FP8.
- Minimize quantization errors in FP8 all-reduce by taking into account the min/max range of participant tensors.
- Perform optimizer state calculations in FP32 to retain precision.


### Tips
- FP8 gives a net positive if the model is large.


- In the FP8 recipe, when you set quantize an input to FP8, that means after an FP8 gemm operation, we will store the outputs as FP8, communicate it using the FP8 format, but if you set an input to float16, that means we store it as float16, but we will also need to quantize it to FP8 in order to performs
