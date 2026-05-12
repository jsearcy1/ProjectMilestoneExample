from pm.dataset.dataloader import get_data_loaders
from pm.train_model import train_model

if __name__ == "__main__":
    train_loader, test_loader, val_loader = get_data_loaders()
    num_classes = 10
    model = train_model(train_loader, val_loader, test_loader, num_classes) 
