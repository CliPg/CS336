import torch
import torch.nn as nn

class rope(nn.Module):

    def __init__(self, theta: float, d_k: int, max_seq_len: int,
                device: torch.device | None = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        R = torch.zeros(max_seq_len, d_k, d_k)
        for i in range(max_seq_len):
            blocks = [self.rotation_block(i, k, theta, d_k) for k in range(d_k // 2)]
            R[i, :, :] = torch.block_diag(*blocks)
        self.R = R
        self.register_buffer("R_BUFFER", self.R, persistent=False)
        
    def rotation_block(self, i: int, k: int, theta: float, d: int):
        angle = torch.tensor(i / (theta ** ((2 * k) / d)))
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        
        return torch.Tensor([[cos, -sin], [sin, cos]])

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        *prefix_dims, seq_len, d_k = x.shape
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)
        R = self.R[token_positions]
        x = R @ x.unsqueeze(-1)
        x = x.squeeze(-1)
        return x