import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes, Pointclouds
from tqdm import tqdm
from metrics.loss import GeometryLoss
import torch.autograd as autograd
import trimesh
from args import parse_args
from model.triplane import NeuSDF, TriPlaneRepresentation, MLPDecoder

def set_globals(args):
    global num_on_surface_samples, num_off_surface_samples
    num_on_surface_samples = args.num_on_surface_points
    num_off_surface_samples = args.num_off_surface_points

def normalize_mesh(mesh):
    verts = mesh.verts_packed()
    center = (verts.max(0)[0] + verts.min(0)[0]) / 2
    scale = (verts - center).abs().max()
    verts = (verts - center) / scale
    mesh = mesh.update_padded(verts.unsqueeze(0))
    return mesh

def sample_sdf_values(mesh):
    '''
    Values inside mesh are positive, values outside mesh are negative
    '''
    # Sample points on the surface of the mesh
    surface_samples, surface_normals = sample_points_from_meshes(mesh, num_on_surface_samples, return_normals=True)
    surface_samples = surface_samples.squeeze(0)
    surface_normals = surface_normals.squeeze(0)
    # Pad surface_normals to match the shape of surface_samples
    zeros_tensor = torch.zeros(num_off_surface_samples, 3).to(device="cuda:0") # Pad zero normal vectors for off_surface points
    surface_normals = torch.cat([surface_normals, zeros_tensor], dim=0)
    # Ground truth SDF values for on-surface points
    surface_sdf_values = torch.zeros(num_on_surface_samples, 1)

    # Convert pytorch3d Mesh to Trimesh
    mesh_trimesh = trimesh.Trimesh(mesh.verts_packed().cpu().numpy(), mesh.faces_packed().cpu().numpy()) # Convert pytorch3d mesh to Trimesh object
    off_surface_samples = (torch.rand(num_off_surface_samples, 3) * 2 - 1) # Uniformly sample points in the [-1, 1]^3 space

    # Compute the SDF values for the off-surface points
    off_surface_sdf_values = torch.tensor(mesh_trimesh.nearest.signed_distance(off_surface_samples)).to(device="cuda:0").unsqueeze(1)
    
    # Back to torch tensors and cuda
    off_surface_samples = off_surface_samples.to(device="cuda:0")
    surface_sdf_values = surface_sdf_values.to(device="cuda:0")

    # Concatenate surface and off-surface samples and their SDF values
    samples = torch.cat([surface_samples, off_surface_samples], dim=0)
    sdf_values = torch.cat([surface_sdf_values, off_surface_sdf_values], dim=0)

    return samples, sdf_values, surface_normals


def preprocess_mesh(mesh_path):
    mesh = load_objs_as_meshes([mesh_path], device='cpu')
    mesh_normalized = normalize_mesh(mesh)
    return mesh_normalized

def get_training_data(mesh):
    points, sdf_values, normal_vectors = sample_sdf_values(mesh)
    return points, sdf_values, normal_vectors

def train_single_object(mesh_path, model_save_path, num_epochs, batch_size, learning_rate, model_mlp, args=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mesh = preprocess_mesh(mesh_path).to(device) # This returns a normalized mesh
    points, sdf_values, normal_vectors = sample_sdf_values(mesh)

    points_tensor = torch.tensor(points, dtype=torch.float32, requires_grad=True).to(device)
    sdf_values_tensor = torch.tensor(sdf_values, dtype=torch.float32).to(device)
    normal_vectors_tensor = torch.tensor(normal_vectors, dtype=torch.float32).to(device)
    dataset = TensorDataset(points_tensor, sdf_values_tensor, normal_vectors_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model_triplane = TriPlaneRepresentation(args.resolution).to(device)
    optimizer = torch.optim.Adam(list(model_triplane.parameters()) + list(model_mlp.parameters()), lr=learning_rate)
    geometry_loss_fn = GeometryLoss()
    loss_log = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            points_batch, sdf_values_batch, normal_vectors_batch = batch
            optimizer.zero_grad()
            pred_features = model_triplane(points_batch)
            pred_sdf = model_mlp(pred_features)

            # Compute the gradient of the sdf prediction w.r.t. the input points
            pred_sdf_grad = autograd.grad(outputs=pred_sdf, inputs=points_batch, grad_outputs=torch.ones_like(pred_sdf), create_graph=True)[0]

            loss = geometry_loss_fn(pred_sdf, pred_sdf_grad, points_batch, sdf_values_batch, normal_vectors_batch)
            # print(f'batch loss = {loss}')
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_epoch_loss = epoch_loss / len(data_loader)
        loss_log.append(avg_epoch_loss)
        print(f"Epoch {epoch+1}, Average Loss: {avg_epoch_loss}")

    torch.save({
        'tri_planes': model_triplane.state_dict(),
        'mlp': model_mlp.state_dict()
    }, model_save_path)

    return loss_log

def train_all_objects(input_dir, output_dir, num_epochs, batch_size, learning_rate, args=None):
    os.makedirs(output_dir, exist_ok=True)
    # Init MLP
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_mlp = MLPDecoder(1, args.mlp_hidden_dim).to(device)
    for root, _, files in os.walk(input_dir):
        if 'model_watertight.obj' in files:
            obj_path = os.path.join(root, 'model_watertight.obj')
            relative_path = os.path.relpath(root, input_dir)
            model_save_path = os.path.join(output_dir, relative_path, 'neusdf_model.pth')
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            loss_log = train_single_object(obj_path, model_save_path, num_epochs, batch_size, learning_rate, model_mlp, args)

            loss_log_path = os.path.join(output_dir, relative_path, 'loss_log.npy')
            np.save(loss_log_path, loss_log)

if __name__ == '__main__':
    argv = None  # Or replace with sys.argv[1:] if you want to pass command line arguments
    args = parse_args(argv)

    # Tech debt :(
    set_globals(args)

    input_dir = args.shapenet_dir
    output_dir = args.triplane_dir
    num_epochs = args.max_epochs
    batch_size = args.train_batch_size
    learning_rate = args.learning_rate

    # TODO: Pass all triplane and mlp dimension args into the trainer loop
    train_all_objects(input_dir, output_dir, num_epochs, batch_size, learning_rate, args)
