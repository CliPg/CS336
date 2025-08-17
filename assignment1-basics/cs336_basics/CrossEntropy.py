import torch
import numpy as np

def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    target_logits = inputs.gather(dim=-1, index=targets.unsqueeze(-1))
    log_sum_exp = torch.logsumexp(inputs, -1, keepdim=True)
    loss_matrix = -target_logits + log_sum_exp
    loss = torch.mean(loss_matrix)
    return loss