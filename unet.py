# Old U-net implementation, not suitable for mc images due to single pixel features
# U-net architecture largely from "Diffusion models beat GANs on image generation"

import torch
import torch.nn as nn
import math
import einops

# torch.backends.cudnn.benchmark = True
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_channels = None,img_dimension_change = None,
                 locally_connected=False,skip = None,diffusion = False):
        super().__init__()
        self.skip = skip
        self.diffusion = diffusion
        if diffusion:
            self.time_emb = nn.Sequential(
                nn.Linear(emb_channels, emb_channels),
                nn.SiLU(),
                nn.Linear(emb_channels, out_channels * 2)
            )
        if img_dimension_change == "upsample":
            self.x_scale = nn.UpsamplingNearest2d(scale_factor=2)
        elif img_dimension_change == "downsample":
            self.x_scale = nn.AvgPool2d(2)
        else:
            self.x_scale = nn.Identity()

        self.conv_block1 = nn.Sequential(
            nn.GroupNorm(num_groups=in_channels//8, num_channels=in_channels),
            nn.SiLU(),
            nn.AvgPool2d(2) if img_dimension_change == "downsample" else nn.Identity(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )


        self.group_norm = nn.GroupNorm(num_groups=out_channels//8, num_channels=out_channels)

        self.conv_block2 = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.UpsamplingNearest2d(scale_factor=2) if img_dimension_change == "upsample" else nn.Identity()
        )

        self.skip_connection = nn.Conv2d(in_channels, out_channels,1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, emb):

        hidden = self.conv_block1(x)
        if self.diffusion:
            scale, bias = self.time_emb(emb)[:, :, None, None].chunk(2, dim=1)
            hidden = self.group_norm(hidden) * (1 + scale) + bias
        else:
            hidden = self.group_norm(hidden)
        hidden = self.conv_block2(hidden)
        x = self.skip_connection(self.x_scale(x))
        return hidden + x


class SelfAttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.MultiheadAttention(channels, num_heads=4, )
        self.to_attn = nn.Conv2d(channels, channels * 3, kernel_size=3,padding=1)
        self.group_norm = nn.GroupNorm(num_groups=channels//8, num_channels=channels)
        self.out = nn.Sequential(
            nn.GroupNorm(num_groups=channels//8, num_channels=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def forward(self, x):
        b, c, w, h = x.shape
        q, k, v = einops.rearrange(self.to_attn(x), 'b c h w -> b (h w) c').chunk(3, dim=-1)
        atn = self.attention(q, k, v)[0].reshape(b, c, w, h) + x
        return self.out(atn) + atn


class Unet(nn.Module):
    def __init__(self, base_channels, diffusion_timesteps, attention_layers=(1, 2, 3)):
        super().__init__()
        time_embed_channels = base_channels * 4
        self.groups = base_channels // 8
        self.sinusodal_encoding_tensor = torch.nn.Parameter(self.sinusodal_encoding(time_embed_channels, diffusion_timesteps+1))
        self.to_time_emb = nn.Sequential(nn.Linear(time_embed_channels, time_embed_channels*4),
                                         nn.SiLU(),
                                         nn.Linear(time_embed_channels*4, time_embed_channels)
                                         )
        self.base_unet = []
        self.first = nn.Conv2d(4, base_channels, 3, padding=1)
        for i in range(3):
            in_channels = base_channels * (2 ** i)
            self.base_unet.extend([
                ResidualBlock(in_channels, in_channels, time_embed_channels),
                SelfAttentionBlock(in_channels) if i in attention_layers else nn.Identity(),
                ResidualBlock(in_channels, in_channels, time_embed_channels,skip = 'out'),
                ResidualBlock(in_channels, in_channels * 2, time_embed_channels, img_dimension_change="downsample")
            ])
        middle_channels = base_channels * 8
        self.base_unet.extend([
            ResidualBlock(middle_channels, middle_channels, time_embed_channels),
            SelfAttentionBlock(middle_channels),
            ResidualBlock(middle_channels, middle_channels, time_embed_channels)
        ])
        for i in range(2, -1, -1):
            in_channels = base_channels * (2 ** i)
            self.base_unet.extend([
                ResidualBlock(in_channels * 2, in_channels, time_embed_channels, img_dimension_change="upsample"),
                ResidualBlock(in_channels * 2, in_channels, time_embed_channels,skip = 'in'),
                SelfAttentionBlock(in_channels) if i in attention_layers else nn.Identity(),
                ResidualBlock(in_channels, in_channels, time_embed_channels),
            ])
        self.base_unet = nn.ModuleList(self.base_unet)
        self.final = nn.Sequential(
            nn.GroupNorm(num_groups=self.groups, num_channels=base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, 4, 3, padding=1),
        )

    def forward(self, x, t):
        time_embedding = self.to_time_emb(self.sinusodal_encoding_tensor[t])
        u_net_skips = []
        x = self.first(x)
        for layer in self.base_unet:
            if isinstance(layer, ResidualBlock):
                if layer.skip == 'out':
                    u_net_skips.append(x)
                elif layer.skip == 'in':
                    j = u_net_skips.pop()
                    x = torch.cat((x, j), dim=1)
                x = layer(x, time_embedding)
            else:
                x = layer(x)
        return self.final(x)

    @staticmethod
    def sinusodal_encoding(d_model, length):
        """
        https://github.com/wzlxjtu/PositionalEncoding2D/blob/d1714f29938970a4999689255f2c230a0b63ebda/positionalembedding2d.py#L5
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

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
