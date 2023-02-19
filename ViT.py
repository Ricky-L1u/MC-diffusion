import torch
import torch.nn as nn
import math

# TODO add bottleneck

class TransformerEncoderBlock(nn.Module):
    def __init__(self, n_dim, n_heads=16):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_dim)
        self.attn = nn.MultiheadAttention(n_dim, num_heads=n_heads, dropout=0.0, batch_first=True)
        self.ln2 = nn.LayerNorm(n_dim, elementwise_affine=False)
        self.proj_out = nn.Sequential(
            nn.Linear(n_dim, n_dim * 4),
            nn.SiLU(),
            nn.Linear(n_dim * 4, n_dim),
        )

    def forward(self, x, scale, shift):
        to_attn = self.ln1(x)
        attn = self.attn(to_attn, to_attn, to_attn)[0] + x
        return self.proj_out(self.ln2(attn) * (scale + 1) + shift) + attn


class ViT(nn.Module):
    def __init__(self, n_dim, diffusion_timesteps, img_size=64, patch_size=4, n_layers=12, cls_token=False):
        super().__init__()
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.n_patches = (img_size // patch_size) ** 2
        flattened_dim = 4 * patch_size * patch_size + int(cls_token)

        self.model_in = nn.Linear(flattened_dim, n_dim)
        self.ViT = nn.ModuleList([TransformerEncoderBlock(n_dim) for _ in range(n_layers)])
        self.model_out = nn.Linear(n_dim,flattened_dim)

        self.embeddings = nn.Parameter(torch.randn(1, self.n_patches, n_dim))
        self.register_buffer('sinusodal_encoding_tensor', self.sinusodal_encodings(n_dim // 4, diffusion_timesteps + 1))
        self.time_proj = nn.Sequential(
            nn.Linear(n_dim // 4, n_dim // 4),
            nn.SiLU(),
            nn.Linear(n_dim // 4, n_dim * n_layers * 2),
        )

    def forward(self, x, t):
        time_proj = self.time_proj(self.sinusodal_encoding_tensor[t]).unsqueeze(1).chunk(self.n_layers * 2, dim=2)
        x = self.model_in(x) + self.embeddings
        for index, layer in enumerate(self.ViT):
            x = layer(x, time_proj[index * 2], time_proj[index * 2 + 1])
        return self.model_out(x)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def sinusodal_encodings(d_model, length):
        # https://github.com/wzlxjtu/PositionalEncoding2D/blob/d1714f29938970a4999689255f2c230a0b63ebda/positionalembedding2d.py#L5
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                              -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe


if __name__ == '__main__':
    j = ViT(n_dim=1024, diffusion_timesteps=1000)
    print(j.num_params)
    print(j(torch.randn(2, 256, 64), torch.LongTensor([0, 1])).shape)
