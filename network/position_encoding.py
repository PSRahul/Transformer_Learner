import torch.nn as nn
import torch
import math
import matplotlib.pyplot as plt


class PositionalEncoding(nn.Module):

    def __init__(self, input_dimension, max_sequence_length=5000):
        super().__init__()

        # It must be of size [Number of Sequences, Hidden Dimension]

        positional_embedding = torch.zeros(
            max_sequence_length, input_dimension)
        position = torch.arange(0, max_sequence_length, dtype=torch.float)
        position = position.unsqueeze(1)
        denominator_term = torch.exp(torch.arange(
            0, input_dimension, 2).float()*(-math.log(10000.0)/input_dimension))

        positional_embedding[:, 0::2] = torch.sin(position*denominator_term)
        positional_embedding[:, 1::2] = torch.cos(position*denominator_term)

        positional_embedding = positional_embedding.unsqueeze(0)

        self.register_buffer('positional_embedding',
                             positional_embedding, persistent=False)

    def forward(self, x):
        x += self.positional_embedding[:, :x.size(1)]
        return x
