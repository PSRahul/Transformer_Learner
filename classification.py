import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from network.ViT_lightning_module import Vision_Transformer_Module
from data.cifar_dataset_class import CIFARDataset
import sys
import os
sys.path.append("/home/psrahul/MasterThesis/repo/ViT_Learner/")


def train_models(**kwargs):
    cifar_dataset = CIFARDataset()
    trainer = pl.Trainer(
        default_root_dir="/home/psrahul/MasterThesis/repo/ViT_Learner/checkpoints/",
        gpus=1,
        max_epochs=10,
        callbacks=[ModelCheckpoint(dirpath="/home/psrahul/MasterThesis/repo/ViT_Learner/checkpoints/models/",
                                   monitor="val_acc", mode="max"),
                   LearningRateMonitor("epoch")],
        progress_bar_refresh_rate=1)

    model = Vision_Transformer_Module(**kwargs)
    trainer.fit(model, cifar_dataset.get_train(), cifar_dataset.get_val())

    val_result = trainer.test(model, cifar_dataset.get_val())
    test_result = trainer.test(model, cifar_dataset.get_test())
    print("Validation Result", val_result)
    print("Test Result", test_result)


def main():

    train_models(model_kwargs={
        'embedding_dim': 256,
        'hidden_dim': 512,
        'num_heads': 8,
        'num_layers': 6,
        'patch_size': 4,
        'num_channels': 3,
        'num_patches': 64,
        'num_classes': 10,
        'dropout': 0.2
    },
        lr=3e-4)


if __name__ == "__main__":
    main()
