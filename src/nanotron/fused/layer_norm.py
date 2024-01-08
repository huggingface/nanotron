from flash_attn.ops.triton.layer_norm import layer_norm_fn
from torch.nn import LayerNorm

class TritonLayerNorm(LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_norm_fn = layer_norm_fn

    def forward(self, input, residual=None, dropout_p=0.0, prenorm=False, residual_in_fp32=False, return_dropout_mask=False):
        return self.layer_norm_fn(input, self.weight, self.bias, residual=residual, eps=self.eps, dropout_p=dropout_p, prenorm=prenorm, residual_in_fp32=residual_in_fp32, is_rms_norm=False, return_dropout_mask=return_dropout_mask)
