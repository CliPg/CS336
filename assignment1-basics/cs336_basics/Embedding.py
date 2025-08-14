import torch
import torch.nn as nn

class Embedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, 
                device: torch.device | None = None, 
                dtype: torch.dtype | None = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        embedding_matrix = torch.empty(num_embeddings, embedding_dim)
        mean = 0
        std = 1
        nn.init.trunc_normal_(embedding_matrix, mean, std, -3*std, 3*std)
        self.embedding_matrix = nn.Parameter(embedding_matrix)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = token_ids.shape
        out_put = torch.empty(batch_size, seq_len, self.embedding_dim)

        for i, seq in enumerate(token_ids):
            for j, token_id in enumerate(seq):
                out_put[i][j] = self.embedding_matrix[token_id]
        
        return out_put


