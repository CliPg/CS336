import torch
import torch.nn as nn
import numpy as np

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        W = torch.empty(out_features, in_features)
        mean = 0
        std = np.sqrt(2 / in_features + out_features)
        nn.init.trunc_normal_(W, mean, std, -3*std, 3*std)
        self.W = nn.Parameter(W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.T