import torch
from torch import nn
import torch.nn.functional as F
import math
from timm.models.layers import trunc_normal_

from models.head import _FCNHead
from models.vig import *

from mobile_sam import sam_model_registry

def autopad(kernel_size):
    return (kernel_size - 1) // 2


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, autopad(kernel_size), bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, autopad(dw_size), groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class EFEM(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, stride=2):
        super().__init__()
        self.ghost1 = GhostModule(inp, int(inp * 2), kernel_size)
        self.convdw = nn.Conv2d(in_channels=int(inp * 2),
                                out_channels=int(inp * 2),
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=autopad(kernel_size),
                                groups=int(inp * 2))
        self.bn = nn.BatchNorm2d(int(inp * 2))
        self.ghost2 = GhostModule(int(inp * 2), oup, kernel_size, stride=1)
        self.shortcut = nn.Sequential(
            nn.Conv2d(inp, inp, kernel_size, stride,
                      autopad(kernel_size), groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(),
            nn.Conv2d(inp, oup, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(),
        )

    def forward(self, x):
        residual = x
        x = self.ghost1(x)
        x = self.convdw(x)
        x = self.bn(x)
        x = self.ghost2(x)
        x += self.shortcut(residual)
        return x

class SAEncoder(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, stride=2, graph=True):
        super().__init__()
        if graph:
            self.conv = Grapher(inp, drop_path=0.0, mode='graph_vig')
        else:
            self.conv = nn.Conv2d(inp, inp, kernel_size, stride=1, padding=autopad(kernel_size), bias=False)
        self.bn = nn.BatchNorm2d(inp)
        self.convdw = nn.Conv2d(in_channels=inp,
                                out_channels=oup,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=autopad(kernel_size))
        
        self.dwbn = nn.BatchNorm2d(oup)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.convdw(x)
        x = self.dwbn(x) #
        x = F.relu(x)    #
        return x


class SADecoder(nn.Module):
    def __init__(self, hidden, oup, kernel_size=3, graph=False):
        super().__init__()
        if graph:
            self.ghost = Grapher(hidden)
        else:
            self.ghost = nn.Conv2d(hidden, hidden, 3, 1, 1, bias=False)
        self.conv = nn.Conv2d(hidden, oup, 1, 1, 0, bias=False)
        self.BN = nn.BatchNorm2d(oup)
        self.act = nn.ReLU()

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True)  # 1,256,256
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.ghost(x1)
        x1 = self.conv(x1)
        x1 = self.BN(x1)
        x1 = self.act(x1)
        return x1

class BPGA(nn.Module):
    def __init__(self, n_class, type="torch"):
        super().__init__()
        self.topk = 4
        self.layer1 = EFEM(3, 128)
        self.layer2 = SAEncoder(128, 128, graph=True)
        self.layer3 = SAEncoder(128, 256, graph=True)
        self.layer4 = SAEncoder(256, 256, graph=True)
        self.decode3 = SADecoder(256 + 256, 256)
        self.decode2 = SADecoder(128 + 256, 128)
        self.decode1 = SADecoder(128 + 128, 128)
        self.head = _FCNHead(128, n_class)

        model_type = "vit_t"

        device = "cuda" if torch.cuda.is_available() else "cpu"

        mobile_sam = sam_model_registry[model_type](checkpoint="mobile_sam.pt")
        mobile_sam.to(device=device)
        
        self.model = mobile_sam.image_encoder

        for param in self.model.parameters():
            param.requires_grad = False

        self.type = type

        self.apply(self.__init_weights)

    def __init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.2)
    
    def sam(self, input):
        h, w = input.shape[-2:]
        padh = self.model.img_size - h
        padw = self.model.img_size - w
        img = F.pad(input, (0, padw, 0, padh))

        if self.type == "torch":
            masks = self.model(img)
        else:
            preds = self.model.infer(img.cpu().numpy())
            masks = torch.from_numpy(preds[0])
        return masks

    def maskencoder(self, input):
        e1 = self.layer01(input)  # 64,128,128
        return e1


    def forward(self, input):
        img = input.detach()  # B C  H W
        masks = self.sam(img)
        _, _, h, w = input.size()
        e1 = self.layer1(input) 
        # mask4, mask3, mask2, mask1 = self.maskencoder(masks.to(input.device))
        e2 = self.layer2(e1)  
        e3 = self.layer3(e2)  
        e4 = self.layer4(e3) 
        # mask1 = self.maskencoder(masks)
        mask1 = F.interpolate(masks, size=e4.shape[2:], mode='bilinear', align_corners=True)
        e4 = e4 + self.cross_attention(e4, mask1[:e4.shape[0], ...])
        d3 = self.decode3(e4, e3) 

        d2 = self.decode2(d3, e2)  

        d1 = self.decode1(d2, e1)  

        out = F.interpolate(d1, size=[h, w], mode='bilinear', align_corners=True)  # 1,256,256
        out = self.head(out)
        return out

    def cross_attention(self, q, k): # k , q
        batch_size, channels, height, width = q.data.size()
        num = channels // k.shape[1]
        q_ = q.view(batch_size, num, k.shape[1], height, width)
        
        """
        交叉注意力机制
        """
        res = []
        for i in range(num):
            q = q_[:, i, ...]
            bs, c, h, w = q.shape
            q = F.adaptive_avg_pool2d(q, (1, 1)).view(bs, c, -1)  # bs, c
            k_ = F.adaptive_avg_pool2d(k, (1, 1)).view(bs, c, -1)  # bs, c
            qk = torch.matmul(q, k_.permute(0, 2, 1)) * c**(-0.5)  # bs, c, c1
            qk = torch.matmul(qk, k_)  # bs, c, h*w 
            qk = qk.view(bs, c, 1, 1)
            qk = F.interpolate(qk, size=[h, w], mode='bilinear', align_corners=True)  # 1,256,256
            res.append(qk)
        qk = torch.cat(res, dim=1)
        qk = F.gelu(qk)
        return qk


if __name__ == "__main__":
    model = BPGA(1)
    from torchsummary import summary
    
    summary(model, (3, 256, 256), device='cpu')
