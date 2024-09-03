import torch
from torch.optim.optimizer import Optimizer

from nanotron import constants
from nanotron.helpers import compute_tensor_stats


class CustomAdamW(Optimizer):
    """Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_

    .. _Adam\\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "amsgrad": False}
        super().__init__(params, defaults)

        self.loggings = {}

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    def _get_optim_logs(self):
        from nanotron.helpers import convert_logs_to_flat_logs

        optim_loggings = {}
        for p in self.loggings:
            param_name = self.params_id_to_param_names[id(p)]
            optim_loggings[param_name] = self.loggings[p]
        return convert_logs_to_flat_logs(optim_loggings)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        loggings = {}

        for group in self.param_groups:
            for p in group["params"]:
                loggings[p] = {}

                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad.pow(2)

                bias_correction1 = 1 - (beta1 ** state["step"])
                bias_correction2 = 1 - (beta2 ** state["step"])

                exp_avg_hat = exp_avg / bias_correction1
                exp_avg_sq_hat = exp_avg_sq / bias_correction2

                normalized_grad = exp_avg_hat / (exp_avg_sq_hat.sqrt() + group["eps"])
                normalized_grad_without_adam_eps = exp_avg_hat / (exp_avg_sq_hat.sqrt())
                weight_decay_grad = group["weight_decay"] * p.data

                new_weight_changes_from_grad = group["lr"] * normalized_grad
                new_weight_changes_from_weight_decay = group["lr"] * weight_decay_grad

                total_new_weight_changes = new_weight_changes_from_grad + new_weight_changes_from_weight_decay

                # LOGGING
                p_norm = p.data.norm()
                weight_norm_and_normalized_grad_update_norm_ratio = p_norm / new_weight_changes_from_grad.norm()
                weight_norm_and_weight_decay_update_norm_ratio = p_norm / new_weight_changes_from_weight_decay.norm()
                weight_norm_and_total_weight_update_norm_ratio = p_norm / total_new_weight_changes.norm()

                p.data = p.data - total_new_weight_changes

                state["exp_avg"] = exp_avg
                state["exp_avg_sq"] = exp_avg_sq

        if constants.is_ready_to_log is True:
            loggings[p]["step"] = {"value": state["step"]}

            loggings[p]["group:lr"] = {"value": group["lr"]}
            loggings[p]["group:eps"] = {"value": group["eps"]}
            loggings[p]["group:beta1"] = {"value": beta1}
            loggings[p]["group:beta2"] = {"value": beta2}

            loggings[p]["bias_correction1"] = {"value": bias_correction1}
            loggings[p]["bias_correction2"] = {"value": bias_correction2}

            loggings[p]["exp_avg"] = compute_tensor_stats(exp_avg)
            loggings[p]["exp_avg_sq"] = compute_tensor_stats(exp_avg_sq)
            loggings[p]["exp_avg_hat"] = compute_tensor_stats(exp_avg_hat)
            loggings[p]["exp_avg_sq_hat"] = compute_tensor_stats(exp_avg_sq_hat)

            loggings[p]["normalized_grad"] = compute_tensor_stats(normalized_grad)
            loggings[p]["normalized_grad_without_adam_eps"] = compute_tensor_stats(normalized_grad_without_adam_eps)
            loggings[p]["weight_decay_grad"] = compute_tensor_stats(weight_decay_grad)

            loggings[p]["fp32_p"] = compute_tensor_stats(p.data)
            loggings[p]["fp32_new_changes_in_p"] = compute_tensor_stats(total_new_weight_changes)
            loggings[p]["fp32_new_changes_from_grad"] = compute_tensor_stats(new_weight_changes_from_grad)

            loggings[p]["fp32_grad"] = compute_tensor_stats(grad)
            loggings[p]["weight_norm_and_normalized_grad_update_norm_ratio"] = {
                "value": weight_norm_and_normalized_grad_update_norm_ratio
            }
            loggings[p]["weight_norm_and_total_weight_update_norm_ratio"] = {
                "value": weight_norm_and_total_weight_update_norm_ratio
            }

            if group["weight_decay"] != 0:
                loggings[p]["fp32_new_changes_from_weight_decay"] = compute_tensor_stats(
                    new_weight_changes_from_weight_decay
                )
                loggings[p]["weight_norm_and_weight_decay_update_norm_ratio"] = {
                    "value": weight_norm_and_weight_decay_update_norm_ratio
                }

        if constants.is_ready_to_log is True:
            self.loggings = loggings
            self.loggings = self._get_optim_logs()

        return loss
