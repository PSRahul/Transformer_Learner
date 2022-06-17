from turtle import forward, position
from unicodedata import decimal
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
        position = torch.arange(input_dimension, dtype=torch.float).ravel()
        denominator_term = torch.exp(torch.arange(
            0, input_dimension))*(-math.log(10000)/input_dimension)

        positional_embedding[:, 0::2] = torch.sin(position*denominator_term)
        positional_embedding[:, 1::2] = torch.cos(position*denominator_term)

        positional_embedding = positional_embedding.unsqueeze(0)

        self.register_buffer('positional_embedding',
                             positional_embedding, persistent=False)

    def forward(self, x):
        x += self.positional_embedding[:, :x.size(1)]


encod_block = PositionalEncoding(48, 96)
pe = encod_block.pe.squeeze().T.cpu().numpy()

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 3))
pos = ax.imshow(pe, cmap="RdGy", extent=(1, pe.shape[1]+1, pe.shape[0]+1, 1))
fig.colorbar(pos, ax=ax)
ax.set_xlabel("Position in sequence")
ax.set_ylabel("Hidden dimension")
ax.set_title("Positional encoding over hidden dimensions")
ax.set_xticks([1]+[i*10 for i in range(1, 1+pe.shape[1]//10)])
ax.set_yticks([1]+[i*10 for i in range(1, 1+pe.shape[0]//10)])
plt.show()
