import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from torchvision import models


class ImageClassifier(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=1e-3):
        super().__init__()

        self.model = models.resnet50(pretrained=True)

        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

        self.save_hyperparameters()

        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.train_accuracy(preds, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.val_accuracy(preds, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.test_accuracy(preds, y)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=5,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


def train_model(train_loader, val_loader, test_loader, num_classes, max_epochs=10):
    model = ImageClassifier(num_classes=num_classes)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=10,
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    return model


if __name__ == "__main__":
    from pm.dataset.dataloader import get_data_loaders

    train_loader, test_loader, val_loader = get_data_loaders()
    num_classes = 10
    model = train_model(train_loader, val_loader, test_loader, num_classes)
