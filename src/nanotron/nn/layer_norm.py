import torch
from flash_attn.ops.triton.layer_norm import layer_norm_fn
from torch import nn


# ###### copy from flash attention1
# import dropout_layer_norm

# def layer_norm(x, weight, bias, epsilon):
#     return DropoutAddLayerNormFn.apply(x, None, weight, bias, None, None, 0.0, epsilon, False)

# class DropoutAddLayerNormFn(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x0, residual, gamma, beta, rowscale, colscale, dropout_p, epsilon,
#                 residual_in_fp32=False, prenorm=False, is_rms_norm=False, return_dmask=False):
#         x0 = maybe_align(x0.contiguous(), 16)
#         residual = maybe_align(residual.contiguous(), 16) if residual is not None else None
#         gamma = maybe_align(gamma.contiguous(), 16)
#         beta = maybe_align(beta.contiguous(), 16) if beta is not None else None
#         rowscale = maybe_align(rowscale.contiguous(), 16) if rowscale is not None else None
#         colscale = maybe_align(colscale.contiguous(), 16) if colscale is not None else None
#         zmat, xmat, dmask, mu, rsigma = _dropout_add_layer_norm_forward(
#             x0, residual, gamma, beta, rowscale, colscale, dropout_p, epsilon,
#             residual_in_fp32, is_rms_norm
#         )
#         # Only need to save x0 if we need to compute gradient wrt colscale
#         x0_saved = x0 if colscale is not None else None
#         ctx.save_for_backward(xmat.view(x0.shape), x0_saved, dmask, gamma, mu, rsigma, rowscale, colscale)
#         ctx.prenorm = prenorm
#         ctx.dropout_p = dropout_p
#         ctx.has_residual = residual is not None
#         ctx.is_rms_norm = is_rms_norm
#         ctx.has_beta = beta is not None
#         if not return_dmask:
#             return (zmat.view(x0.shape) if not prenorm
#                     else (zmat.view(x0.shape), xmat.view(x0.shape)))
#         else:
#             dmask = (dmask.view(x0.shape) if dropout_p > 0.
#                      else torch.ones(x0.shape, dtype=torch.uint8, device=x0.device))
#             ctx.mark_non_differentiable(dmask)
#             return ((zmat.view(x0.shape), dmask) if not prenorm
#                     else (zmat.view(x0.shape), xmat.view(x0.shape), dmask))

#     @staticmethod
#     def backward(ctx, dz, *args):
#         # assert dz.is_contiguous()
#         dz = maybe_align(dz.contiguous(), 16)  # this happens!
#         dx = maybe_align(args[0].contiguous(), 16) if ctx.prenorm else None
#         x, x0, dmask, gamma, mu, rsigma, rowscale, colscale = ctx.saved_tensors
#         # x0 is None if colscale is None
#         dropout_p = ctx.dropout_p
#         has_residual = ctx.has_residual
#         dx0mat, dresidualmat, dgamma, dbeta, *rest = _dropout_add_layer_norm_backward(
#             dz, dx, x, x0, dmask, mu, rsigma, gamma, rowscale, colscale, dropout_p, has_residual,
#             ctx.is_rms_norm
#         )
#         dx0 = dx0mat.view(x.shape)
#         dresidual = dresidualmat.view(x.shape) if dresidualmat is not None else None
#         dcolscale = rest[0] if colscale is not None else None
#         return (dx0, dresidual, dgamma, dbeta if ctx.has_beta else None, None, dcolscale, None,
#                 None, None, None, None, None)

# def maybe_align(x, alignment_in_bytes=16):
#     """Assume that x already has last dim divisible by alignment_in_bytes
#     """
#     # TD [2023-07-04] I'm not 100% sure that clone will align the memory
#     # https://discuss.pytorch.org/t/how-to-ensure-that-tensor-data-ptr-is-aligned-to-16-bytes/183440
#     return x if x.data_ptr() % alignment_in_bytes == 0 else x.clone()

# def _dropout_add_layer_norm_forward(x0, residual, gamma, beta, rowscale, colscale, dropout_p,
#                                     epsilon, residual_in_fp32=False, is_rms_norm=False):
#     """ Assume that arguments are contiguous and aligned to 16 bytes
#     """
#     hidden_size = gamma.numel()
#     x0mat = x0.view((-1, hidden_size))
#     residualmat = residual.view((-1, hidden_size)) if residual is not None else None
#     rowscale = rowscale.view(-1) if rowscale is not None else None
#     zmat, xmat, dmask, mu, rsigma = dropout_layer_norm.dropout_add_ln_fwd(
#         x0mat, residualmat, gamma, beta, rowscale, colscale, None, None, dropout_p, epsilon,
#         1.0, 0, None, residual_in_fp32, is_rms_norm
#     )
#     # dmask is None if dropout_p == 0.0
#     # xmat is None if dropout_p == 0.0 and residual is None and residual_dtype != input_dtype
#     return zmat, xmat if xmat is not None else x0mat, dmask, mu, rsigma

# def _dropout_add_layer_norm_backward(dz, dx, x, x0, dmask, mu, rsigma, gamma, rowscale, colscale,
#                                      dropout_p, has_residual, is_rms_norm=False):
#     """ Assume that arguments are contiguous and aligned to 16 bytes
#     dx == None means that it was a post-norm architecture
#     (x = drop(x0) + residual was not returned in the fwd).
#     x0 must not be None if we have colscale.
#     """
#     hidden_size = gamma.numel()
#     xmat = x.view((-1, hidden_size))
#     dzmat = dz.view(xmat.shape)
#     dxmat = dx.view(xmat.shape) if dx is not None else None
#     x0mat = x0.view((-1, hidden_size)) if x0 is not None else None
#     rowscale = rowscale.view(-1) if rowscale is not None else None
#     if colscale is not None:
#         assert x0 is not None, 'x0 is required to compute the gradient of colscale'
#     dx0mat, dresidualmat, dgamma, dbeta, _, _, *rest = dropout_layer_norm.dropout_add_ln_bwd(
#         dzmat, dxmat, xmat, x0mat, dmask, mu, rsigma, gamma, rowscale, colscale, None, None,
#         dropout_p, 1.0, 0, has_residual, is_rms_norm
#     )
#     # dresidualmat is None if not has_residual
#     if colscale is None:
#         return dx0mat, dresidualmat, dgamma, dbeta
#     else:
#         dcolscale = rest[0]
#         return dx0mat, dresidualmat, dgamma, dbeta, dcolscale

# ######



class TritonLayerNorm(nn.LayerNorm):
    def forward(
        self, input, residual=None, dropout_p=0.0, prenorm=False, residual_in_fp32=False, return_dropout_mask=False
    ):
        ## test
        # return layer_norm(
        #     x=input,
        #     weight=self.weight,
        #     bias=self.bias,
        #     epsilon=self.eps,
        # )
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
        return layer_norm_fn(
            input,
            self.weight,
            None,
            residual=residual,
            eps=self.eps,
            dropout_p=dropout_p,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
            is_rms_norm=True,
            return_dropout_mask=return_dropout_mask,
        )


# residual None
# self.eps 1e-05
# dropout_p 0.0
# prenorm False
# residual_in_fp32 False
# is_rms_norm=True,
# return_dropout_mask False

## pas important
# x1=None,
# weight1=None,
# bias1=None,
# rowscale=None,