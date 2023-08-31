import torch
import torch.nn as nn

from lib.LSG.LSG import LSG_iter, sparse_LSG_iter
from AConv import AConv
from Self_attention import SA

def conv_bn_relu(in_c, out_c):
    return nn.Sequential(
        AConv(in_c, out_c, 7, 5, 3, stride=1, dilation=1),
        # nn.Conv2d(in_c, out_c, 1, padding=0, bias=False),
        # SA(in_c, out_c,out_c//5, 5, 9),
        nn.BatchNorm2d(out_c),
        nn.ReLU(True)
    )

class LSGModel(nn.Module):
    def __init__(self, feature_dim, nspix, bands, n_iter=10):
        super().__init__()
        self.nspix = nspix
        self.n_iter = n_iter
        self.bands = bands

        self.scale1 = nn.Sequential(
            conv_bn_relu(self.bands, 32)
        )
        self.scale2 = nn.Sequential(
            conv_bn_relu(32, 32)
        )
        self.scale3 = nn.Sequential(
            conv_bn_relu(64, 32)
        )

        self.output_conv = nn.Sequential(
            nn.Conv2d(96, feature_dim-5, 1, padding=0),
            nn.ReLU(True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        

    def forward(self, x):
        pixel_f = self.feature_extract(x)

        if self.training:
            return LSG_iter(pixel_f, self.nspix, self.n_iter)
        else:
            return sparse_LSG_iter(pixel_f, self.nspix, self.n_iter)


    def feature_extract(self, x):
        s1 = self.scale1(x)
        s2 = self.scale2(s1)
        s3 = torch.cat([s1, s2], 1)
        s3 = self.scale3(s3)
        s4 = torch.cat([s1, s2, s3], 1)

        feat = self.output_conv(s4)

        return torch.cat([feat, x], 1)
