import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split


class DataPipeline:

    def __init__(self, batch_size=128, data_dir='./animals10_data/raw-img', image_size=128):
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.image_size = image_size

        self.train_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

    def create_dataloaders(self, train_ratio=0.7, val_ratio=0.2):
        full_dataset = torchvision.datasets.ImageFolder(
                root=f'{self.data_dir}',
                transform=self.train_transform
            )

        total_size = len(full_dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size

        print(f"Data split:")
        print(f"  Training: {train_size} images")
        print(f"  Validation: {val_size} images")
        print(f"  Test: {test_size} images")

        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        val_dataset.dataset.transform = self.test_transform
        test_dataset.dataset.transform = self.test_transform

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )

        return train_loader, val_loader, test_loader
