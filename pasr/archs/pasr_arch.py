from torch import nn as nn
from torch.nn import functional as F

from basicsr.archs.arch_util import default_init_weights
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class PASR(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, upscale=4):
        super(PASR, self).__init__()
        self.upscale = upscale
        pass

    def forward(self, x):
        out = x
        return out
