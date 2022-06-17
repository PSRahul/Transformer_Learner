from ast import Mult
import imp
from turtle import forward
import torch.nn as nn
from network.multihead import MultiHeadAttention


class EncoderBlock(nn.Module):

    def __init__(self, input_dim, num_heads, feedforward_dim, dropout=0):
        super().__init__()

        # Pass through the Multi-Head Attention
        self.multihead_attention_layer = MultiHeadAttention(
            input_dim, input_dim, num_heads)

        self.linear_layer = nn.Sequential(
            nn.Linear(input_dim, feedforward_dim),
            nn.Dropout(dropout),
            nn.SELU(inplace=True),
            nn.Linear(feedforward_dim, input_dim)
        )

        self.layernorm_1 = nn.LayerNorm(input_dim)
        self.layernorm_2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):

        # The Attention Block
        attention_output_value = self.multihead_attention_layer(x, mask)
        x += self.dropout(attention_output_value)
        x = self.layernorm_1(x)

        # Feed Forward Layer

        linear_output = self.linear_layer(x)
        x += self.dropout(linear_output)
        x = self.layernorm_2(x)

        return x
