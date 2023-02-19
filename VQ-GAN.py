# For future VQ-GAN (unused)

import torch
import torch.nn as nn
import einops

from unet import ResidualBlock, SelfAttentionBlock


class Encoder(nn.Module):
    def __init__(self, base_channels, f, codebook_dims):
        super().__init__()
        self.in_conv = nn.Conv2d(4, base_channels, 3, padding=1)
        compression = []
        for i in range(0, f):
            in_channels = base_channels * (2 ** i)
            out_channels = in_channels * 2
            compression.extend((
                ResidualBlock(in_channels, out_channels),
                nn.AvgPool2d(2)
            ))
        self.compression = nn.Sequential(*compression)
        out_channels = base_channels * (2 ** f)
        self.final = nn.Sequential(
            ResidualBlock(out_channels, out_channels),
            SelfAttentionBlock(out_channels),
            ResidualBlock(out_channels, out_channels),
            nn.GroupNorm(num_groups=32, num_channels=out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, codebook_dims, 3, padding=1)
        )

    def forward(self, x):
        return self.in_conv(self.compression(self.final(x)))


class StraightThroughEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, codebook):
        closest_index = torch.argmin(torch.sum((x - codebook) ** 2, dim=3), dim=2)
        return codebook[:, :, closest_index, :].squeeze()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class Codebook(nn.Module):
    def __init__(self, codebook_emb_dims, codebook_lookup_dims, num_codebooks):
        super().__init__()
        self.codebook_lookup_dims = codebook_lookup_dims
        self.codebook = nn.Parameter(torch.randn(1, 1, num_codebooks, codebook_lookup_dims))
        self.proj_in = nn.Linear(codebook_emb_dims, codebook_lookup_dims)
        self.proj_out = nn.Linear(codebook_lookup_dims, codebook_emb_dims)
        self.ste = StraightThroughEstimator.apply

    def forward(self, x):
        # TODO jitscript
        b, c, h, w = x.shape
        to_codebook = einops.rearrange(x, 'b (c n) h w -> b (h w) n c', n=1)
        to_codebook = self.proj_in(to_codebook)
        nearest = self.ste(to_codebook, self.codebook)
        out = self.proj_out(nearest)
        return einops.rearrange(out, 'b (h w) c -> b c h w', h=h, w=w), nearest


class VQGAN(nn.Module):
    def __init__(self, base_channels, f, codebook_dims):
        super().__init__()
        self.encoder = Encoder(base_channels, f, codebook_dims)
        self.codebook = Codebook(codebook_dims, codebook_dims, codebook_dims)
        self.decoder = None
        self.discriminator = None

    def forward(self, x):
        out, nearest_low_dim = self.codebook(self.encoder(x))
        return self.decoder(out), nearest_low_dim

    def train_step(self, x, data):
        pass

if __name__ == "__main__":
    j = Codebook(2, 3, 4)
    x = torch.randn(6, 2, 5, 5)
    print(j(x)[0].shape)
