import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.ops import sample_points_from_meshes
import os
import torch

class ShapeNetDataset(Dataset):
    def __init__(self, data_dir, num_samples_on=5000, num_samples_off=5000, sigma=0.01):
        self.data_dir = data_dir
        self.mesh_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.obj')]
        self.num_samples_on = num_samples_on
        self.num_samples_off = num_samples_off
        self.sigma = sigma

    def __len__(self):
        return len(self.mesh_files)

    def __getitem__(self, idx):
        mesh_path = self.mesh_files[idx]
        mesh = load_objs_as_meshes([mesh_path])
        
        on_surface_points, on_surface_normals = sample_points_from_meshes(mesh, self.num_samples_on, return_normals=True)
        off_surface_points = self.sample_off_surface_points(mesh, self.num_samples_off, self.sigma)
        
        return on_surface_points, on_surface_normals, off_surface_points

    def sample_off_surface_points(self, mesh, num_samples, sigma):
        sampled_points, _ = sample_points_from_meshes(mesh, num_samples)
        noise = sigma * torch.randn_like(sampled_points)
        off_surface_points = sampled_points + noise
        return off_surface_points

class NeuSDFDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=8, num_samples_on=5000, num_samples_off=5000, sigma=0.01):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_samples_on = num_samples_on
        self.num_samples_off = num_samples_off
        self.sigma = sigma

    def setup(self, stage=None):
        dataset = ShapeNetDataset(self.data_dir, self.num_samples_on, self.num_samples_off, self.sigma)
        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
