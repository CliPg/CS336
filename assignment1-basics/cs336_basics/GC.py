import torch
from collections.abc import Iterable

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    parameters_with_grad = [p for p in parameters if p.grad is not None]

    if len(parameters) == 0:
        return
    
    norm = torch.sqrt(sum(torch.sum(p.grad.pow(2)) for p in parameters_with_grad))
    eps = 1e-6
    clip_coef = max_l2_norm / (norm + eps)

    if norm > max_l2_norm:
        for p in parameters_with_grad:
            p.grad.data.mul_(clip_coef)

        