import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from einops import rearrange

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding"""
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()

        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        return x

class MRConv4d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type

    K is the number of superpatches, therefore hops equals res // K.
    """

    def __init__(self, in_channels, out_channels, K=2):
        super(MRConv4d, self).__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1),
            nn.BatchNorm2d(in_channels * 2),
            nn.GELU()
        )
        self.K = K

    def forward(self, x):
        B, C, H, W = x.shape
        x = torch.roll(x, shifts=(-(self.K // 2), -(self.K // 2)), dims=(2, 3))
        x_j = torch.zeros_like(x).to(x.device)
        for i in torch.arange(self.K, H, self.K):
            x_c = x - torch.roll(x, shifts=(-i, 0), dims=(2, 3))
            x_j = torch.max(x_j, x_c)
        for i in torch.arange(self.K, W, self.K):
            x_r = x - torch.roll(x, shifts=(0, -i), dims=(2, 3))
            x_j = torch.max(x_j, x_r)

        x = torch.cat([x, x_j], dim=1)
        x = torch.roll(x, shifts=(self.K // 2, self.K // 2), dims=(2, 3))
        return self.nn(x)

class MRConv4dTF(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type

    K is the number of superpatches, therefore hops equals res // K.
    """

    def __init__(self, in_channels, out_channels, K=2):
        super(MRConv4dTF, self).__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1),
            nn.BatchNorm2d(in_channels * 2),
            nn.GELU()
        )
        self.K = K

    def forward(self, x):
        B, C, H, W = x.shape
        x_j = torch.zeros_like(x).to(x.device)
        
        q = rearrange(x, 'b c h w -> b (h w) c')
        k = q.clone()
        v = q.clone()
        att = torch.matmul(q, k.transpose(-2, -1))
        att = F.softmax(att, dim=-1)
        att = torch.matmul(att, v)
        att = rearrange(att, 'b (h w) c -> b c h w', h=H, w=W)

        x_r = x - att
        x_j = torch.max(x_j, x_r)

        x = torch.cat([x, x_j], dim=1)
        return self.nn(x)

class MRConv4dSP(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type

    K is the number of superpatches, therefore hops equals res // K.
    """

    def __init__(self, in_channels, out_channels, K=2):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1),
            nn.BatchNorm2d(in_channels * 2),
            nn.GELU()
        )

        self.d  = in_channels ** -0.5

    # def forward(self, x):
    #     x_j = torch.zeros_like(x).to(x.device)
    #     x_j[:, :, ::4, ::4] = 1
    #     select = x[:, :, ::4, ::4]
    #     B, C, H, W = select.shape
    #     q = rearrange(select, 'b c h w -> b (h w) c')
    #     k = q.clone()
    #     v = q.clone()
    #     att = torch.matmul(q, k.transpose(-2, -1))* self.d
    #     att = F.softmax(att, dim=-1)
    #     att = torch.matmul(att, v)
    #     att = rearrange(att, 'b (h w) c -> b c h w', h=H, w=W)
    #     att = F.interpolate(att, x_j.shape[2:], mode='nearest')
    #     att = att * x_j
    #     x = torch.cat([x, att], dim=1)
    #     return self.nn(x)

    def forward(self, x): # 2024/11/6 修改
        x_j = torch.zeros_like(x).to(x.device)
        x_j[:, :, ::4, ::4] = 1
        select = x[:, :, ::4, ::4]
        B, C, H, W = select.shape
        q = rearrange(select, 'b c h w -> b (h w) c')
        k = q.clone()
        v = q.clone()
        att = torch.matmul(q, k.transpose(-2, -1))* self.d
        att = F.softmax(att, dim=-1)
        att = torch.matmul(att, v)
        att = rearrange(att, 'b (h w) c -> b c h w', h=H, w=W)
        att = F.interpolate(att, x_j.shape[2:], mode='nearest')
        # att = att * x_j
        x = torch.cat([x, att], dim=1)
        return self.nn(x)

class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """

    def __init__(self, in_channels, drop_path=0.0, K=2, mode='graph_vig'):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.K = K

        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
        )
        
        if mode == 'mobile_vig':
            self.graph_conv = MRConv4d(in_channels, in_channels * 2, K=self.K)
        elif mode == 'graph_vig':
            self.graph_conv = MRConv4dSP(in_channels, in_channels * 2, K=self.K)

        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
        )  
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        _tmp = x
        x = self.fc1(x)
        x = self.graph_conv(x)
        x = self.fc2(x)
        x = self.drop_path(x) + _tmp

        return x