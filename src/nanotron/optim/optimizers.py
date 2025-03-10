import torch
from torch import Tensor

############################
#            Muon          #
############################

# code mainly from https://github.com/KellerJordan/modded-nanogpt and https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py


@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert (
        G.ndim >= 2
    )  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750, 2.0315)  # TODO @eliebak: do some more ablation on this
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = (
            b * A + c * A @ A
        )  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        param_groups: The list of parameter groups.
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """

    def __init__(self, param_groups, lr, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = {"lr": lr, "momentum": momentum, "nesterov": nesterov, "ns_steps": ns_steps}

        # Handle param_groups directly
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Update rule:
        1. m_t = β·m_{t-1} + (1-β)·g_t         # momentum buffer
        2. u_t = m_t or β·m_t + g_t            # standard or Nesterov
        3. u'_t = NS(u_t)                      # orthogonalization
        4. θ_t = θ_{t-1}·(1-lr·λ) - lr·u'_t    # weight decay + update

        where NS() is Newton-Schulz orthogonalization, β is momentum,
        λ is weight decay, and g_t is gradient.
        """
        if closure is not None:
            with torch.enable_grad():
                closure()

        for group in self.param_groups:
            lr = group.get(
                "lr", self.defaults["lr"]
            )  # TODO @eliebak: make sure the learning rate is already be scaled here
            weight_decay = group.get("weight_decay", 0.0)
            # TODO @eliebak: not sure momentum/nesterov/ns_steps are important to have in the param groups, figure out the level of granularity we want here.
            momentum = group.get("momentum", self.defaults["momentum"])
            nesterov = group.get("nesterov", self.defaults["nesterov"])
            ns_steps = group.get("ns_steps", self.defaults["ns_steps"])

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buf = state["momentum_buffer"]
                buf.lerp_(grad, 1 - momentum)
                grad = grad.lerp_(buf, momentum) if nesterov else buf

                # Orthogonalization step
                grad = zeropower_via_newtonschulz5(grad, steps=ns_steps)

                # Apply weight decay directly to parameters
                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)

                # Parameter update with the orthogonalized gradient
                p.add_(grad, alpha=-lr)
