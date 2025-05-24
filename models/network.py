import torch.nn as nn
import torch.nn.functional as F

from models.model import *


def get_BPGA(num_class=1):
    return BPGA(num_class)


if __name__ == '__main__':
    model = get_BPGA(1)
    from torchsummary import summary
    summary(model, (3, 256, 256), device='cpu')
