import torch
import torch.nn as nn
from einops import einsum, rearrange
from .RoPE import rope
from .Linear import Linear

def softmax(x: torch.Tensor, dim: int):
        x_max = torch.max(x, dim, keepdim=True).values
        exp_x = torch.exp(x - x_max)
        sum_exp_x = torch.sum(exp_x, dim, keepdim=True)
        
        out_put = exp_x / sum_exp_x
        return out_put

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None):
        d_k = Q.shape[-1]
        attention = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / (d_k ** 0.5)

        if mask is not None:
            attention = attention.masked_fill(~mask, float('-inf'))

        attention = softmax(attention, dim=-1)
        
        out_put = attention @ V
        return out_put

class multihead_self_attention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, use_rope: bool = False, max_seq_len: int | None = None, 
                 theta: float | None = None, token_positions: torch.Tensor | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_rope = use_rope
        self.rope = rope(theta, d_model // num_heads, max_seq_len) if use_rope else None
        self.token_positions = token_positions
        self.Qproj = Linear(d_model, d_model)
        self.Kproj = Linear(d_model, d_model)
        self.Vproj = Linear(d_model, d_model)
        self.MSA = Linear(d_model, d_model)
    
    def split_heads(self, x: torch.Tensor):
        return rearrange(x, "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim",
                         num_heads = self.num_heads, head_dim = self.head_dim
                        )

    def forward(self, in_features: torch.Tensor):
        seq_len = in_features.shape[-2]
        
        qkv_proj = torch.cat([self.Qproj.W, self.Kproj.W, self.Vproj.W])
        qkv = in_features @ qkv_proj.T
        Q, K, V = qkv.chunk(3, -1)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        if self.use_rope:
            Q = self.rope(Q, self.token_positions)
            K = self.rope(K, self.token_positions)
        
        casual_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        casual_mask = casual_mask[None, None, :, :]

        multi_head = scaled_dot_product_attention(Q, K, V, ~casual_mask)
        multi_head = rearrange(multi_head, "... num_heads seq_len d -> ... seq_len (num_heads d)")

        multi_head_self_attention = self.MSA(multi_head)
        return multi_head_self_attention

        


