import torch


class NeuSDF(torch.nn.Module):
    def __init__(self, triplane_resolution, mlp_hidden_dim):
        super(NeuSDF, self).__init__()
        self.triplane = TriPlaneRepresentation(triplane_resolution)
        self.mlp = MLPDecoder(1, mlp_hidden_dim)  # Input dimension is 1 since the sum of features results in a scalar

    def forward(self, points):
        features = self.triplane(points)
        sdf_values = self.mlp(features)
        return sdf_values
    

class TriPlaneRepresentation(torch.nn.Module):
    def __init__(self, resolution):
        super(TriPlaneRepresentation, self).__init__()
        self.xy_plane = torch.nn.Parameter(torch.randn(resolution, resolution))
        self.yz_plane = torch.nn.Parameter(torch.randn(resolution, resolution))
        self.xz_plane = torch.nn.Parameter(torch.randn(resolution, resolution))
        self.resolution = resolution

    def forward(self, points):
        # Normalize the points to the grid resolution
        points = (points + 1) / 2 * (self.resolution - 1)
        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        def interpolate(plane, coord1, coord2):
            coord1_floor = torch.floor(coord1).long()
            coord2_floor = torch.floor(coord2).long()
            coord1_ceil = torch.clamp(coord1_floor + 1, max=self.resolution - 1)
            coord2_ceil = torch.clamp(coord2_floor + 1, max=self.resolution - 1)

            bl = plane[coord1_floor, coord2_floor]
            br = plane[coord1_ceil, coord2_floor]
            tl = plane[coord1_floor, coord2_ceil]
            tr = plane[coord1_ceil, coord2_ceil]

            h_lerp = coord1 - coord1_floor.float()
            v_lerp = coord2 - coord2_floor.float()

            top = tl + (tr - tl) * h_lerp
            bottom = bl + (br - bl) * h_lerp
            return bottom + (top - bottom) * v_lerp

        # Interpolation on each plane
        Fxy = interpolate(self.xy_plane, x, y)
        Fyz = interpolate(self.yz_plane, y, z)
        Fxz = interpolate(self.xz_plane, x, z)

        # Element-wise addition of interpolated values from each plane
        F = Fxy + Fxz + Fyz
        F = F.unsqueeze(-1)
        return F


class MLPDecoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(MLPDecoder, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim), # output_dim = 1 bc we are mapping triplane features to a single scalar distance
            torch.nn.Tanh() # Constrain output to [-1, 1]
        )

    def forward(self, features):
        # Features should already be a flat vector from the sum in TriPlaneRepresentation
        sdf_values = self.mlp(features)
        return sdf_values
    
