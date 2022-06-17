import re
from utils.math_utils import *
import torch.nn as nn


class MultiHeadAttention(nn.Module):

    def __init__(self, input_dim, embedding_dim, num_heads):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = self.embedding_dim//self.num_heads

        # Convert input dimension to Query, Key and Values Shapes
        self.input_projection_layer = nn.Linear(input_dim, 3*embedding_dim)
        self.output_projection_layer = nn.Linear(embedding_dim, embedding_dim)

        self.initialize()

    def initialize(self):
        nn.init.xavier_uniform_(self.input_projection_layer.weight)
        nn.init.xavier_uniform_(self.output_projection_layer.weight)

        self.input_projection_layer.bias.data.fill_(0)
        self.output_projection_layer.bias.data.fill_(0)

    def forward(self, input_batch, mask=None, return_attention=False):

        # Input Projection Layer
        batch_size, sequence_length, key_dimension = input_batch.size()
        attention_keys = self.input_projection_layer(input_batch)
        # Shape -> Batch Size, Sequence Length, Num Heads, Attention Keys *3
        attention_keys = attention_keys.reshape(
            batch_size, sequence_length, self.num_heads, 3*self.head_dim)
        # Shape -> Batch Size, Num Heads, Sequence Length,  Attention Keys *3
        attention_keys = attention_keys.permute(0, 2, 1, 3)
        query, key, value = attention_keys.chunk(3, dim=3)
        # We get the Query, Key and Value for each of the Heads

        # Output Projection Layer

        output_value, output_attention = scaled_dot_product(
            query, key, value, mask)
        # Shape -> Batch Size, Sequence Length, Num Heads, Attention Keys *3
        output_value = output_value.permute(0, 2, 1, 3)
        # Shape -> Batch Size, Sequence Length, Num Heads* Attention Keys *3
        output_value = self.output_projection_layer(output_value)

        if return_attention:
            return output_value, output_attention
        else:
            return output_value
