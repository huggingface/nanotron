import torch
from torch import nn


class TritonLayerNorm(nn.LayerNorm):
    def forward(
        self, input, residual=None, dropout_p=0.0, prenorm=False, residual_in_fp32=False, return_dropout_mask=False
    ):
        from flash_attn.ops.triton.layer_norm import layer_norm_fn

        return layer_norm_fn(
            input,
            self.weight,
            self.bias,
            residual=residual,
            eps=self.eps,
            dropout_p=dropout_p,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
            is_rms_norm=False,
            return_dropout_mask=return_dropout_mask,
        )


# This is equivalent to LLaMA RMSNorm
# https://github.com/huggingface/transformers/blob/28952248b19db29ca25ccf34a5eec413376494a9/src/transformers/models/llama/modeling_llama.py#L112
class TritonRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)

    def forward(
        self, input, residual=None, dropout_p=0.0, prenorm=False, residual_in_fp32=False, return_dropout_mask=False
    ):
        # NOTE: fa=2.6.3
        # got the following errors:
        # Traceback (most recent call last):
        # File "/fsx/phuc/temp/fp8_for_nanotron/env/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
        #     return self._call_impl(*args, **kwargs)
        # File "/fsx/phuc/temp/fp8_for_nanotron/env/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
        #     return forward_call(*args, **kwargs)
        # File "/fsx/phuc/temp/fp8_for_nanotron/nanotron/src/nanotron/nn/layer_norm.py", line 44, in forward
        #     return layer_norm_fn(
        # File "/fsx/phuc/temp/fp8_for_nanotron/env/lib/python3.10/site-packages/flash_attn/ops/triton/layer_norm.py", line 875, in layer_norm_fn
        #     return LayerNormFn.apply(
        # File "/fsx/phuc/temp/fp8_for_nanotron/env/lib/python3.10/site-packages/torch/autograd/function.py", line 539, in apply
        #     return super().apply(*args, **kwargs)  # type: ignore[misc]
        # File "/fsx/phuc/temp/fp8_for_nanotron/env/lib/python3.10/site-packages/flash_attn/ops/triton/layer_norm.py", line 748, in forward
        #     y, y1, mean, rstd, residual_out, seeds, dropout_mask, dropout_mask1 = _layer_norm_fwd(
        # File "/fsx/phuc/temp/fp8_for_nanotron/env/lib/python3.10/site-packages/flash_attn/ops/triton/layer_norm.py", line 335, in _layer_norm_fwd
        #     _layer_norm_fwd_1pass_kernel[(M,)](
        # File "/fsx/phuc/temp/fp8_for_nanotron/env/lib/python3.10/site-packages/triton/runtime/jit.py", line 345, in <lambda>
        #     return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
        # File "/fsx/phuc/temp/fp8_for_nanotron/env/lib/python3.10/site-packages/triton/runtime/autotuner.py", line 156, in run
        #     timings = {config: self._bench(*args, config=config, **kwargs) for config in pruned_configs}
        # File "/fsx/phuc/temp/fp8_for_nanotron/env/lib/python3.10/site-packages/triton/runtime/autotuner.py", line 156, in <dictcomp>
        #     timings = {config: self._bench(*args, config=config, **kwargs) for config in pruned_configs}
        # File "/fsx/phuc/temp/fp8_for_nanotron/env/lib/python3.10/site-packages/triton/runtime/autotuner.py", line 133, in _bench
        #     return do_bench(kernel_call, warmup=self.num_warmups, rep=self.num_reps, quantiles=(0.5, 0.2, 0.8))
        # File "/fsx/phuc/temp/fp8_for_nanotron/env/lib/python3.10/site-packages/triton/testing.py", line 104, in do_bench
        #     torch.cuda.synchronize()
        # File "/fsx/phuc/temp/fp8_for_nanotron/env/lib/python3.10/site-packages/torch/cuda/__init__.py", line 783, in synchronize
        #     return torch._C._cuda_synchronize()
        # RuntimeError: CUDA error: an illegal memory access was encountered
        # CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
        # For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
        # Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

        # from flash_attn.ops.triton.layer_norm import layer_norm_fn
        # return layer_norm_fn(
        #     input,
        #     self.weight,
        #     None,
        #     residual=residual,
        #     eps=self.eps,
        #     dropout_p=dropout_p,
        #     prenorm=prenorm,
        #     residual_in_fp32=residual_in_fp32,
        #     is_rms_norm=True,
        #     return_dropout_mask=return_dropout_mask,
        # )

        # NOTE: fa=2.4.2
        from flash_attn.ops.triton.layernorm import rms_norm_fn

        return rms_norm_fn(
            input,
            self.weight,
            None,
            residual=residual,
            eps=self.eps,
            dropout_p=dropout_p,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
            # is_rms_norm=True, # NOTE: fa=2.4.2 don't use this? wtf dao
            return_dropout_mask=return_dropout_mask,
        )
