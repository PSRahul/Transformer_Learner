# %%
import torch
import math
import torch.nn.functional as F


# %%
def scaled_dot_product(query, key, value, mask=None):
    d_k = query.size()[-1]
    attention_logits = torch.matmul(query, key.transpose(-2, -1))
    attention_logits = attention_logits/math.sqrt(d_k)
    if mask is not None:
        attention_logits.masked_fill_(mask == 0, 1e-15)

    attention = F.softmax(attention_logits, dim=-1)
    value = torch.matmul(attention, value)
    return value, attention
