# U-NET needs to be adapted for this generative task, as opposed to image segmentation
# TODO: Implement U-Net as Lightning Module
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.attend = nn.Softmax(dim=-1)

    def forward(self, x, context):
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        return out

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # U-NET Encoder
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # U_NET Decoder
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        # Cross-attention layer
        self.cross_attention = CrossAttention(dim=features[0])

    def forward(self, x, condition):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            # Cross-attention before concatenation
            attn_out = self.cross_attention(x, condition)
            x = torch.cat((attn_out, skip_connection), dim=1)
            x = self.ups[idx + 1](x)

        return self.final_conv(x)

# Example usage
# model = UNet(in_channels=3, out_channels=1)
# x = torch.randn((1, 3, 256, 256))  # Example input image
# condition = torch.randn((1, 64, 64, 64))  # Example condition
# out = model(x, condition)
