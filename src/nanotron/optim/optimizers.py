import math

import torch

############################
#            Muon          #
############################

# code mainly from https://github.com/KellerJordan/modded-nanogpt and https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py


@torch.compile(fullgraph=True)
def nsloop_torch(X: torch.Tensor, steps: int, *, a=3.4445, b=-4.7750, c=2.0315):
    """
    When compiled down, inductor produces the following steps:
    1. A = matmul X with reinterpret_tensor(X)
    2. (triton) read A -> write b*A and c*A
    3. B = addmm(b*A, c*A, A)
    4. (triton) read X -> write a*X (this is stupid)
    5. X = addmm(a*X, B, X)
    """
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X


def zeropower_via_newtonschulz5(G, steps, f_iter=nsloop_torch):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2, f"Shape of G is {G.shape}"
    X = G.to(dtype=torch.bfloat16)
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    X = f_iter(X, steps)

    if G.size(0) > G.size(1):
        X = X.T
    return X


# code from https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py
class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        wd=0.1,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
        spectral_mup_scaling=False,
        moonlight_scaling=True,
        sign_muon=False,
    ):

        defaults = {
            "lr": lr,
            "wd": wd,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "adamw_betas": adamw_betas,
            "adamw_eps": adamw_eps,
        }
        self.sign_muon = sign_muon
        self.moonlight_scaling = moonlight_scaling
        self.spectral_mup_scaling = spectral_mup_scaling

        assert not (moonlight_scaling and spectral_mup_scaling), "Cannot have both moonlight and spectral mup scaling"

        ## custom nanotron logic
        # what we want here:
        # - muon_params should be everything except 1D parameters (layer norms) and 2D parameters like embeddings and lm_heads
        # - we can handle a more precise granuarility after but i think we don't have the .name of the param groups here (that should be added, for now just have something hardcoded will do the job)
        id_to_name = params["id_to_name"]
        params = params["params"]
        muon_params = []
        adamw_params = []
        for param_group in params:
            muon_param_group = {
                **{k: v for k, v in param_group.items() if k != "params"},
                "params": [],
            }
            adamw_param_group = {
                **{k: v for k, v in param_group.items() if k != "params"},
                "params": [],
            }
            for param in param_group["params"]:
                if any(keyword in id_to_name[id(param)] for keyword in ["embed", "lm_head", "norm"]):
                    adamw_param_group["params"].append(param)
                else:
                    muon_param_group["params"].append(param)
            if len(muon_param_group["params"]) > 0:
                muon_params.append(muon_param_group)
            if len(adamw_param_group["params"]) > 0:
                adamw_params.append(adamw_param_group)

        super().__init__(params, defaults)
        # Sort parameters into those for which we will use Muon, and those for which we will not
        for p_group in muon_params:
            for p in p_group["params"]:
                # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
                assert p.ndim == 2, p.ndim
                self.state[p]["use_muon"] = True
        for p_group in adamw_params:
            for p in p_group["params"]:
                # Do not use Muon for parameters in adamw_params
                self.state[p]["use_muon"] = False

    def adjust_lr_for_muon(self, lr, param_shape):
        if self.moonlight_scaling:
            A, B = param_shape[:2]
            # We adjust the learning rate and weight decay based on the size of the parameter matrix
            # as describted in the paper
            adjusted_ratio = 0.2 * math.sqrt(max(A, B))
            adjusted_lr = lr * adjusted_ratio
        elif self.spectral_mup_scaling:
            # Adjust learning rate based on fan-out/fan-in ratio
            fan_out, fan_in = param_shape[:2]
            adjusted_lr = lr * math.sqrt(max(1, fan_out / fan_in))
        else:
            adjusted_lr = lr
        return adjusted_lr

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            # import pdb; pdb.set_trace()
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

            # generate weight updates in distributed fashion
            for p in params:
                # sanity check
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None

                # calc update
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                if self.sign_muon:
                    g = torch.sign(g)
                u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

                # scale update
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

                # apply weight decay
                p.data.mul_(1 - lr * wd)

                # apply update
                p.data.add_(u, alpha=-adjusted_lr)

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group["lr"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss
