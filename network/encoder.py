import torch.nn as nn

from network.encoder_block import EncoderBlock


class TransformerEncoder(nn.Module):

    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)

        return x

    def get_attention_maps(self, x, mask=None):

        attention_maps = []

        for layer in self.layers:
            _, attention_map = layer.multihead_attention_layer(
                x, mask, return_attention=True)
            attention_maps.append(attention_map)

            x = layer(x)
        return attention_maps
