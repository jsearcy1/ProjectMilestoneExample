import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        print(img_dir)
        self.transform = transform if transform is not None else transforms.ToTensor()

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir, self.data_frame.iloc[idx]['image'])
        image = Image.open(img_name).convert('RGB')
        label = self.data_frame.iloc[idx]['label']

        if self.transform:
            image = self.transform(image)

        return image, label

def get_data_loaders(base_path='/projects/dsci410_510/jakes_example', batch_size=32):
    """
    Create data loaders for train, test, and dev sets
    
    Args:
        base_path (string): Base path to the dataset
        batch_size (int): Batch size for the dataloaders
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = CustomImageDataset(
        csv_file=os.path.join(base_path, 'train.csv'),
        img_dir=base_path,
        transform=transform
    )

    test_dataset = CustomImageDataset(
        csv_file=os.path.join(base_path, 'test.csv'),
        img_dir=base_path,
        transform=transform
    )

    dev_dataset = CustomImageDataset(
        csv_file=os.path.join(base_path, 'develop.csv'),
        img_dir=os.path.join(base_path,'develop'),
        transform=transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    return train_loader, test_loader, dev_loader

# Example usage:
if __name__ == "__main__":
    train_loader, test_loader, dev_loader = get_data_loaders()
    
    # Print sample batch
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        break
