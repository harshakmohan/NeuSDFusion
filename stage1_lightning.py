import os
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch3d.ops import sample_points_from_meshes
from torch.utils.data import DataLoader, Dataset
from pytorch3d.io import load_objs_as_meshes
import trimesh
import torch.autograd as autograd
from model.triplane import NeuSDF
from metrics.loss import GeometryLoss
from args import parse_args


class MeshDataset(Dataset):
    def __init__(self, root_dir, output_dir, num_on_surface, num_off_surface):
        super().__init__()
        self.mesh_files = []
        self.model_save_paths = []
        self.num_on_surface = num_on_surface
        self.num_off_surface = num_off_surface
        
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith("model_watertight.obj"):
                    full_path = os.path.join(subdir, file)
                    self.mesh_files.append(full_path)

                    # Calculate relative path and model save path
                    relative_path = os.path.relpath(subdir, root_dir)
                    model_save_path = os.path.join(output_dir, relative_path, 'neusdf_model.pth')
                    self.model_save_paths.append(model_save_path)

    def __len__(self):
        return len(self.mesh_files)

    def __getitem__(self, idx):
        mesh_path = self.mesh_files[idx]
        model_save_path = self.model_save_paths[idx]

        mesh = load_objs_as_meshes([mesh_path], device='cpu')
        mesh_normalized = self.normalize_mesh(mesh)
        points, sdf_values, normal_vectors = self.sample_sdf_values(mesh_normalized)
        return points, sdf_values, normal_vectors, model_save_path

    @staticmethod
    def normalize_mesh(mesh):
        verts = mesh.verts_packed()
        center = verts.mean(0)
        scale = (verts - center).abs().max()
        verts = (verts - center) / scale
        mesh = mesh.update_padded(verts.unsqueeze(0))
        return mesh

    def sample_sdf_values(self, mesh):
        surface_samples, surface_normals = sample_points_from_meshes(mesh, self.num_on_surface, return_normals=True)
        surface_samples = surface_samples.squeeze(0)
        surface_normals = surface_normals.squeeze(0)
        surface_sdf_values = torch.zeros(self.num_on_surface, 1)

        mesh_trimesh = trimesh.Trimesh(mesh.verts_packed().cpu().numpy(), mesh.faces_packed().cpu().numpy())
        off_surface_samples = (torch.rand(self.num_off_surface, 3) * 2 - 1).to(device="cuda:0")
        off_surface_sdf_values = torch.tensor(mesh_trimesh.nearest.signed_distance(off_surface_samples)).unsqueeze(1)

        samples = torch.cat([surface_samples, off_surface_samples], dim=0).to(device="cuda:0")
        sdf_values = torch.cat([surface_sdf_values, off_surface_sdf_values], dim=0)
        normal_vectors = torch.cat([surface_normals, torch.zeros(self.num_off_surface, 3).to(device="cuda:0")], dim=0)

        return samples, sdf_values, normal_vectors

class MeshDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, output_dir, num_on_surface, num_off_surface):
        super().__init__()
        #TODO: How do we get the save path for the model to use in on_train_end?
        self.dataset = MeshDataset(root_dir, output_dir, num_on_surface, num_off_surface)

    def train_dataloader(self):
        # I don't think this is gonna work. We need multiple epochs per object.
        return DataLoader(self.dataset, batch_size=1, shuffle=False)


class MeshModel(pl.LightningModule):
    def __init__(self, triplane_resolution, mlp_hidden_dim, learning_rate):
        super().__init__()
        self.model = NeuSDF(triplane_resolution, mlp_hidden_dim)
        self.learning_rate = learning_rate
        self.loss_fn = GeometryLoss()
        # TODO: Add save path for the triplanes and mlp
        self.model_save_path = None

    def forward(self, points):
        return self.model(points)

    def on_train_start(self):
        pass
    
    def training_step(self, batch, batch_idx):
        points, sdf_values, normal_vectors, model_save_path = batch
        pred_sdf = self.forward(points)
        pred_sdf_grad = autograd.grad(outputs=pred_sdf, inputs=points, grad_outputs=torch.ones_like(pred_sdf), create_graph=True)[0]
        loss = self.loss_fn(pred_sdf, pred_sdf_grad, points, sdf_values, normal_vectors)
        self.model_save_path = model_save_path
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def on_train_end(self):
        save_path = os.path.join(self.save_dir, 'neusdf_model.pth')
        torch.save({
            'tri_planes': self.model.triplane.state_dict(),
            'mlp': self.model.mlp.state_dict()
        }, save_path)
        print(f"Model saved to {save_path}")

def main():
    args = parse_args(None)
    pl.seed_everything(args.seed)

    data_module = MeshDataModule(args.shapenet_dir, args.triplane_dir, args.num_on_surface_points, args.num_off_surface_points)
    model = MeshModel(args.triplane_resolution, args.mlp_hidden_dim, args.learning_rate)
    trainer = pl.Trainer(max_epochs=args.max_epochs, default_root_dir=args.triplane_dir)
    trainer.fit(model, datamodule=data_module)

if __name__ == '__main__':
    main()
