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

    def __init__(self, d_model: int, num_heads: int, d_k: int, d_v: int, d_in: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.d_k = d_k
        self.d_v = d_v
        self.d_in = d_in
        self.head_dim = d_k // num_heads

        self.Qproj = Linear(d_k, d_in)
        self.Kproj = Linear(d_k, d_in)
        self.Vproj = Linear(d_v, d_in)
        self.MSA = Linear(d_model, d_v)
    
    def split_heads(self, x: torch.Tensor):
        return rearrange(x, "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim",
                         num_heads = self.num_heads, head_dim = self.head_dim
                        )

    def forward(self, in_features: torch.Tensor):
        seq_len = in_features.shape[-2]
        Q = self.Qproj(in_features)
        K = self.Kproj(in_features)
        V = self.Vproj(in_features)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        casual_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        casual_mask = casual_mask[None, None, :, :]

        multi_head = scaled_dot_product_attention(Q, K, V, ~casual_mask)
        multi_head = rearrange(multi_head, "... num_heads seq_len d -> ... seq_len (num_heads d)")

        multi_head_self_attention = self.MSA(multi_head)
        return multi_head_self_attention
    
class multihead_self_attention_with_rope(nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_k: int, d_v: int, d_in: int, 
                 max_seq_len: int, theta: float, token_positions):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.token_positions = token_positions

        self.d_k = d_k
        self.d_v = d_v
        self.d_in = d_in
        self.head_dim = d_k // num_heads

        self.Qproj = Linear(d_k, d_in)
        self.Kproj = Linear(d_k, d_in)
        self.Vproj = Linear(d_v, d_in)
        self.MSA = Linear(d_model, d_v)

        self.rope = rope(theta, self.head_dim, max_seq_len)
    
    def split_heads(self, x: torch.Tensor):
        return rearrange(x, "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim",
                         num_heads = self.num_heads, head_dim = self.head_dim
                        )

    def forward(self, in_features: torch.Tensor):
        seq_len = in_features.shape[-2]
        Q = self.Qproj(in_features)
        K = self.Kproj(in_features)
        V = self.Vproj(in_features)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        Q = self.rope(Q, self.token_positions)
        K = self.rope(K, self.token_positions)

        casual_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        casual_mask = casual_mask[None, None, :, :]

        multi_head = scaled_dot_product_attention(Q, K, V, ~casual_mask)
        multi_head = rearrange(multi_head, "... num_heads seq_len d -> ... seq_len (num_heads d)")

        multi_head_self_attention = self.MSA(multi_head)
        return multi_head_self_attention

        


