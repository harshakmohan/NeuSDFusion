import os
import sys
import torch
import torch.nn as nn
import lightning as pl
from lightning import Trainer
from torch.utils.data import Dataset, DataLoader
from args import parse_args
from model.triplane import NeuSDF, TriPlaneRepresentation, MLPDecoder
import torch.nn.functional as F


class NeuSDFDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.data = []

        mesh_dir = args.shapenet_dir  # Directory containing the meshes
        triplane_dir = args.triplane_dir  # Directory containing the triplanes
        self.triplane_resolution = args.resolution
        self.mlp_hidden_dim = args.mlp_hidden_dim

        for subdir, dirs, files in os.walk(triplane_dir):
            if 'models' in dirs:
                models_dir = os.path.join(subdir, 'models')
                identifier = os.path.basename(subdir)  # Extract the identifier
                mesh_path = os.path.join(mesh_dir, identifier, "models", "model_watertight.obj")
                
                for file in os.listdir(models_dir):
                    if file.endswith('.pth'):
                        full_path = os.path.join(models_dir, file)
                        self.data.append((full_path, mesh_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pth_path, mesh_path = self.data[idx]
        state_dict = torch.load(pth_path)

        triplane = TriPlaneRepresentation(self.triplane_resolution)
        # Loading state dict for triplanes, don't need to worry about the MLP.
        triplane.load_state_dict(state_dict['tri_planes'])
        
        # Concatenate the triplanes into a single tensor
        x = triplane.xy_plane.squeeze(), triplane.yz_plane.squeeze(), triplane.xz_plane.squeeze()
        stacked_triplanes = torch.stack(x, dim=0)
        stacked_triplanes = stacked_triplanes.half()  # Use half precision bc small gpu

        # Load the corresponding mesh
        with open(mesh_path, 'r') as file:
            mesh = file.read()
        return stacked_triplanes, mesh


class NeuSDFDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_batch_size = args.stage2_train_batch_size

    def setup(self, stage=None):
        self.dataset = NeuSDFDataset(self.args)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.train_batch_size, shuffle=True)
    

class ResidualBlock(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(d_model, hidden_dim, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(hidden_dim, d_model, kernel_size=1)
        self.conv_out = nn.Conv3d(d_model, 1, kernel_size=1)
    
    def forward(self, x):
        # x shape: [batch_size, d_model, 3, resolution, resolution]
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual  # Residual connection
        out = self.conv_out(out)  # Mapping to 1 channel
        out = out.squeeze(1)  # Removing the channel dimension
        # Output shape: [batch_size, 3, resolution, resolution]
        return out


class TriplaneAutoEncoder(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args


        ### Encoder Components ###
        embedding_dim = 3 * args.resolution * args.resolution
        self.encoder_embedding = nn.Linear(embedding_dim, embedding_dim)

        self.group_down_conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=args.gc_kernel_size, stride=args.gc_stride, padding=1, groups=3)
        nn.init.kaiming_normal_(self.group_down_conv.weight, mode='fan_out', nonlinearity='relu')
        if self.group_down_conv.bias is not None:
            nn.init.zeros_(self.group_down_conv.bias)

        input_dim = embedding_dim // (args.gc_stride ** 2)
        self.encoder_input_proj = nn.Linear(input_dim, args.d_model)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=args.d_model, nhead=args.n_heads, dim_feedforward=args.d_model*2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=args.num_encoder_layers)

        # Linear Layers to get μ and log(σ)
        self.fc_mu = nn.Linear(args.d_model, args.d_model)
        self.fc_sigma = nn.Linear(args.d_model, args.d_model)

        ### Decoder Components ###
        self.decoder_embedding = nn.Linear(args.d_model, args.d_model)
        self.group_up_conv = nn.ConvTranspose2d(in_channels=args.d_model, out_channels=args.d_model, kernel_size=args.gc_kernel_size, stride=args.gc_stride, padding=1, output_padding=1, groups=args.d_model)
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=args.d_model, 
            nhead=args.n_heads, 
            dim_feedforward=args.d_model*2, 
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=args.num_decoder_layers)
        # Learnable target sequence embedding
        tgt_seq_len = 3 * (args.resolution // args.gc_stride) ** 2
        self.tgt_embed = nn.Parameter(torch.randn(1, tgt_seq_len, args.d_model))
        # Residual MLP Decoder
        self.res_block = ResidualBlock(args.d_model, 128)

    def reparameterize(self, mu, log_sigma):
        if self.training:
            std = torch.exp(0.5 * log_sigma)
            eps = torch.randn_like(std) # Sample eps from gaussian
            return mu + eps * std
        else:
            return mu

    def encode(self, x):
        # Step 1: Flatten triplane into 1D vector [batch size, 3 * height * width]
        x = x.reshape(x.size(0), -1)
        # Step 2: Pass through embedding layer
        x = self.encoder_embedding(x)
        # Step 3: Reshape back to original shape prior to grouped down conv
        x = x.reshape(x.size(0), 3, self.args.resolution, self.args.resolution)
        # Step 4: Grouped Convolution
        x = self.group_down_conv(x) # [batch size, 3, height/2, width/2]
        # TODO: Step 5: Add SAPE to input prior to passing through encoder

        # Step 6: Flatten prior to transformer encoder
        x = x.reshape(x.size(0), -1)
        # Step 7: Pass through transformer encoder
        x = self.encoder_input_proj(x)
        x = self.transformer_encoder(x) # [batch_size, d_model]
        # Step 8: Pass through output layer to get μ and log(σ) per output token
        mu = self.fc_mu(x)
        log_sigma = self.fc_sigma(x)
        # Step 9: Reparameterize
        z = self.reparameterize(mu, log_sigma) # [batch_size, d_model]
        return z

    def decode(self, z):
        # Step 1: Prepare memory and target sequence for transformer decoder
        memory = self.decoder_embedding(z).unsqueeze(1) # [batch_size, 1, d_model]
        tgt = self.tgt_embed.expand(z.size(0), -1, -1) # [batch_size, tgt_seq_len, d_model]
        # Step 2: Pass through transformer decoder
        z = self.transformer_decoder(tgt, memory)
        # Step 3: Reshape prior to Group Up Convolution
        z = z.reshape(z.size(0), 3, self.args.resolution // self.args.gc_stride, self.args.resolution // self.args.gc_stride, self.args.d_model) # [batch size, 3, downsampled_res, downsampled_res, d_model]
        batch_size, num_planes, downsampled_h, downsampled_w, d_model = z.size()
        z = z.permute(0, 1, -1, 2, 3).contiguous()
        z = z.view(batch_size*num_planes, d_model, downsampled_h, downsampled_w)
        # Step 4: Group Up Convolution along d_model dimension
        upsampled = self.group_up_conv(z) # [batch_size*num_planes, d_model, height, width]
        # Step 5: Reshape back to original shape
        upsampled = upsampled.view(batch_size, num_planes, d_model, self.args.resolution, self.args.resolution)
        upsampled = upsampled.permute(0, 2, 1, 3, 4).contiguous() # [batch size, d_model, 3, height, width]
        # Step 6: Pass through Residual MLP Decoder to get reconstructed triplane tokens
        recon_triplane_tokens = self.res_block(upsampled) # [batch size, 3, height, width]
        return recon_triplane_tokens

    def training_step(self, batch, batch_idx):
        # Batch comes as list of len 2. First element is batch of triplanes, second is batch of meshes
        triplanes, meshs = batch # triplanes shape: [batch_size, 3, height, width]
        meshes_hat = self.forward(triplanes)

    def forward(self, x):
        z = self.encode(x)
        recon_triplane_tokens = self.decode(z)
        

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    model = TriplaneAutoEncoder(args)
    model = model.half()  # Use half precision bc small gpu

    data_module = NeuSDFDataModule(args)
    data_module.setup()

    # Pass in necessary args into the trainer
    trainer = Trainer(max_epochs=args.stage2_max_epochs)

    trainer.fit(model, datamodule=data_module)
