import sys
sys.path.append("/home/psrahul/MasterThesis/repo/ViT_Learner/")

from network.nn_attentionblock import AttentionBlock
import torch
from data.image_transforms import image_to_patches
import torch.nn as nn
from data.cifar_dataset_class import CIFARDataset


class VisionTransformer(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, num_channels,
                 num_heads, num_layers, num_classes, patch_size, num_patches, dropout):

        super().__init__()

        self.patch_size = patch_size

        self.input_layer = nn.Linear(
            num_channels * (patch_size**2), embedding_dim)

        self.transformer = nn.Sequential(
            *[AttentionBlock(embedding_dim, hidden_dim, num_heads, dropout)
              for _ in range(num_layers)]
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, num_classes)
        )

        self.dropout = nn.Dropout(dropout)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.pos_embedding = nn.Parameter(
            torch.randn(1, 1 + num_patches, embedding_dim))

        #print("Positional Embedding", self.pos_embedding.shape)
        #print("Class Token", self.cls_token.shape)

    def forward(self, x):
        x = image_to_patches(x, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)
        # Add the class token and the positional embedding

        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x += self.pos_embedding[:, :T + 1]

        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)

        cls = x[0]
        output = self.mlp_head(cls)
        return output


"""
vision_transformer = VisionTransformer(embedding_dim=12, hidden_dim=24, num_channels=3,
                                       num_heads=4, num_layers=2, num_classes=10, patch_size=2,
                                       num_patches=256, dropout=0)

cifar_dataset = CIFARDataset()
train_dataset = cifar_dataset.get_train()
for idx in train_dataset:
    image = idx[0]
    break

x = vision_transformer.forward(image)
"""
