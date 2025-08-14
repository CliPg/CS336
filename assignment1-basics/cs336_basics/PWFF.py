import torch
import torch.nn as nn

class positionwise_feedforward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, 
                device: torch.device | None = None, 
                dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype

        self.W1 = nn.Parameter(torch.empty(d_ff, d_model))
        self.W2 = nn.Parameter(torch.empty(d_model, d_ff))
        self.W3 = nn.Parameter(torch.empty(d_ff, d_model))

    def SiLU(self, x: torch.Tensor):
        return x * torch.sigmoid(x)
    
    def FFN(self, x: torch.Tensor, W1: torch.Tensor, W2: torch.Tensor, W3: torch.Tensor):
        return (self.SiLU(x @ W1.T) * (x @ W3.T)) @ W2.T
    
    def forward(self, x: torch.Tensor):
        return self.FFN(x, self.W1, self.W2, self.W3)
