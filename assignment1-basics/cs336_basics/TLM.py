import torch
import torch.nn as nn
from .Embedding import Embedding
from .TB import transformer_block
from .RMSNorm import rmsnorm
from .Linear import Linear

class transformer_lm(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, 
                 vocab_size: int, context_len: int, num_layers: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        self.vocab_size = vocab_size
        self.context_len = context_len
        self.num_layers = num_layers
        self.embedding_model = Embedding(vocab_size, d_model)
        self.transformer_layers = nn.ModuleList(
            transformer_block(d_model, num_heads, d_ff, context_len, theta)
            for _ in range(num_layers)
        )
        self.rmsnorm_model = rmsnorm(d_model)
        self.out_put_proj = Linear(d_model, vocab_size)

    def forward(self, in_indices: torch.Tensor):
        out_put = self.embedding_model(in_indices)

        for layer in self.transformer_layers:
            out_put = layer(out_put)

        out_put = self.rmsnorm_model(out_put)
        out_put = self.out_put_proj(out_put)
        return out_put