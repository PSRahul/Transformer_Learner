import pytorch_lightning as pl
import network.nn_ViT as VisionTransformer
import torch.optim as optim
import torch.functional as F


class Vision_Transformer_Module(pl.LightningModule):

    def __init__(self, model_kwargs, lr):
        super().__init__()

        self.save_hyperparameters()
        self.model = VisionTransformer(**model_kwargs)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150])
        return [optimizer], [lr_scheduler]

    def calculate_loss(self, batch, mode="train"):
        images, labels = batch
        predictions = self.model(images)
        loss = F.cross_entropy(predictions, labels)
        accuracy = (predictions.argmax(dim=-1) == labels).float().mean()

        self.log(f'{mode}_loss', loss)
        self.log(f'{mode}_acc', accuracy)

    def training_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch=batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch=batch, mode="val")

    def test_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch=batch, mode="test")
