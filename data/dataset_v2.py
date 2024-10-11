import os
import random
import torch
from torch.utils.data import DataLoader, Dataset
import lightning as pl
from pathlib import Path

class ShapeNetDataset(Dataset):
    def __init__(self, root_dir, percentage=1.0, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_paths = list(Path(root_dir).rglob('model_normalized.obj'))
        self.sample_size = int(len(self.file_paths) * percentage)
        self.sampled_paths = random.sample(self.file_paths, self.sample_size)

    def __len__(self):
        return len(self.sampled_paths)

    def __getitem__(self, idx):
        file_path = self.sampled_paths[idx]
        data = self.load_obj(file_path)
        if self.transform:
            data = self.transform(data)
        return data

    def load_obj(self, file_path):
        # Implement this method to load .obj files as required
        # Placeholder for loading the .obj file and returning tensor data
        return torch.tensor([0.0])  # Replace with actual data loading logic

class ShapeNetDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, percentage=1.0, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.percentage = percentage
        self.transform = transform

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = ShapeNetDataset(self.data_dir, percentage=self.percentage, transform=self.transform)
        if stage == 'test' or stage is None:
            self.test_dataset = ShapeNetDataset(self.data_dir, percentage=self.percentage, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

# Usage
data_dir = '/home/harsha/Documents/shapenet/02691156'
batch_size = 32
percentage = 0.5  # Load 50% of the models
transform = None  # Add necessary transformations

shapenet_dm = ShapeNetDataModule(data_dir, batch_size, percentage, transform)

# Example to test the dataloader
shapenet_dm.setup()
train_loader = shapenet_dm.train_dataloader()
for batch in train_loader:
    print(batch)
