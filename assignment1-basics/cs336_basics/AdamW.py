from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class adamw_optimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                
                state["step"] += 1

                m, v = state["m"], state["v"]
                a = group["lr"]
                b1, b2 = group["betas"]
                eps = group["eps"]
                lam = group["weight_decay"]
                step = state["step"]

                state["m"] = b1 * m + (1 - b1) * grad
                state["v"] = b2 * v + (1 - b2) * (grad**2)
                state["a_t"] = a * math.sqrt(1 - b2**step) / (1 - b1**step)

                m = state["m"]
                v = state["v"]
                a_t = state["a_t"]

                p.data = p.data - a_t * m / (v.sqrt() + eps)
                p.data = p.data - a * lam * p.data

        return loss





