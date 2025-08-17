import torch
import torch.nn as nn
from .MSA import multihead_self_attention_with_rope, multihead_self_attention
from .PWFF import positionwise_feedforward
from .RMSNorm import rmsnorm

class transformer_block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.rmsnorm_model1 = rmsnorm(self.d_model)
        self.rmsnorm_model2 = rmsnorm(self.d_model)
        self.msa_model = multihead_self_attention_with_rope(self.d_model, self.num_heads, self.d_model, self.d_model, self.d_model, 
                                                       self.max_seq_len, self.theta, None)
        self.pwff_model = positionwise_feedforward(d_model, d_ff)

    def forward(self, in_features: torch.Tensor):
        out_put = in_features + self.msa_model(self.rmsnorm_model1(in_features))
        out_put = out_put + self.pwff_model(self.rmsnorm_model2(out_put))
        return out_put
