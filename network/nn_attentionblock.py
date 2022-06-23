from this import s
import torch.nn as nn


class AttentionBlock(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_heads, dropout=0):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embedding_dim)
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads,
                                               dropout=dropout)
        self.layer_norm_2 = nn.LayerNorm(embedding_dim)

        self.linear_layer = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.layer_norm_1(x)
        attn_output, attn_output_weights = self.attention(x, x, x)
        x = x+attn_output
        x = x+self.linear_layer(self.layer_norm_2(x))
        return x
