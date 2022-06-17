# %%
import torch
import pytorch_lightning as pl
from utils.math_utils import *

# %%
sequence_length = 3
key_dimension = 2
pl.seed_everything(42)

# %%
query = torch.randn(1, sequence_length, key_dimension)
key = torch.randn(1, sequence_length, key_dimension)
value = torch.randn(1, sequence_length, key_dimension)
output_value, output_attention = scaled_dot_product(query, key, value)


# %%
print("Query\n", query)
print("Key\n", key)
print("Values\n", value)
print("Output Values\n", output_value)
print("Output Attention\n", output_attention)
