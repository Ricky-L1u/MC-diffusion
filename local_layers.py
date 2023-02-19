# Unused local layers (each pixel has its own conv)
# Makes sense to use local conv in context of mc skins but unfortunately it's too slow.
# TODO needs value scaling to ensure output values dont blow up
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair


class LocallyConnected2d(torch.jit.ScriptModule):
    """Adapted from https://discuss.pytorch.org/t/locally-connected-layers/26979/2"""

    def __init__(self, height, in_channels, out_channels, kernel_size, stride=1, bias=False):
        super(LocallyConnected2d, self).__init__()
        out_dims = (height - 3) / stride + 1
        output_size = _pair(out_dims)
        self.unfold = torch.nn.Unfold(3)
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, output_size[0], output_size[1], kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1])
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)

    @torch.jit.script_method
    def forward(self, x):
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        c = torch.einsum('bihwkd,oihwkd->bohw', x, self.weight)
        if self.bias is not None:
            c += self.bias
        return c


class LocallyConnectedLinear(torch.jit.ScriptModule):
    def __init__(self, in_channels, out_channels, pix_dim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_channels, pix_dim, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.randn(1, pix_dim, out_channels))
        else:
            self.register_parameter('bias', None)

    @torch.jit.script_method
    def forward(self, x):
        loc = torch.einsum('bpi,ipo->bpo', x, self.weight)
        return loc + self.bias if self.bias is not None else loc