from re import L
from turtle import forward
import pytorch_lightning as pl
import torch.nn as nn
from network.position_encoding import PositionalEncoding
from network.encoder import TransformerEncoder
from network.cosine_warmup import CosineWarmupScheduler


class Transformer(pl.LightningModule):

    def __init__(self, input_dim, transformer_model_dim, num_class,
                 num_heads, num_layers, learning_rate, warmup, max_iterations,
                 dropout=0, input_dropout=0):

        super().__init__()
        self.save_hyperparameters()
        self.init_model()

    def init_model():

        self.input_processor = nn.Sequential(
            nn.Dropout(self.hparams.input_dropout),
            nn.Linear(self.hparams.input_dim,
                      self.hparams.transformer_model_dim)
        )

        self.positional_encoding = PositionalEncoding(
            input_dimension=self.hparams.transformer_model_dim)

        self.transformer = TransformerEncoder(
            num_layers=self.hparams.num_layers,
            input_dim=self.hparams.transformer_model_dim,
            feedforward_dim=self.hparams.transformer_model_dim,
            num_heads=self.hparams.num_heads,
            dropout=self.hparams.dropout)

        self.output_processor = nn.Sequential(
            nn.Linear(self.hparams.transformer_model_dim,
                      self.hparams.transformer_model_dim),
            nn.LayerNorm(self.hparams.transformer_model_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.transformer_model_dim,
                      self.hparams.num_class)
        )

    def forward(self, batch, mask=None, add_positional_encoding=True):
        # Shape of the Batch - Batch Size, Number of Sequences, Hidden Dimension

        x = self.input_processor(batch)

        if add_positional_encoding:
            x = self.positional_encoding(x)

        x = self.transformer(x)
        x = self.output_processor(x)

        return x

    @torch.no_grad()
    def get_attention_maps(self, batch, mask=None, add_positional_encoding=True):

        x = self.input_processor(batch)

        if add_positional_encoding:
            x = self.positional_encoding(x)

        attention_maps = self.transformer.get_attention_maps(x, mask)

        return attention_maps

    def configure_optimizers(self):

        optimizer = optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate)
        learning_rate_scheduler = CosineWarmupScheduler(
            optimizer, warmup=self.hparams.warmup,
            max_iterations=self.hparams.max_iterations)

        return [optimizer], [{'scheduler': learning_rate_scheduler, 'interval': 'step'}]
