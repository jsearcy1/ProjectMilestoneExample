import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from pm.model.model import ImageClassifier


def train_model(train_loader, val_loader, test_loader, num_classes, max_epochs=10):
    model = ImageClassifier(num_classes=num_classes)

    early_stopping = EarlyStopping(monitor="val_loss", patience=10, mode="min")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=10,
        callbacks=[early_stopping],
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    return model
