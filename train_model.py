import pytorch_lightning as pl
from model.model import ImageClassifier 

def train_model(train_loader, val_loader, test_loader, num_classes, max_epochs=10):
    # Initialize model
    model = ImageClassifier(num_classes=num_classes)

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min',save_top_k=1)
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='auto',  # Automatically detect if you have GPU
        devices=1,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=10,
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader, callbacks=[early_stopping])
    
    # Test model
    trainer.test(model, test_loader)
    
    return model

if __name__ == "__main__":
    # Example usage
    from dataset.dataloader import get_data_loaders
    
    # Get data loaders
    train_loader, test_loader, val_loader = get_data_loaders()
    
    # Assuming you have 10 classes, modify this based on your dataset
    num_classes = 10
    
    # Train the model
    model = train_model(train_loader, val_loader, test_loader, num_classes) 