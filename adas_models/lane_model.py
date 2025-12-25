import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class LaneSegNet(nn.Module):
    def __init__(self, backbone="mobilenetv2_100", out_ch=1, pretrained=False):
        super().__init__()
        self.enc = timm.create_model(backbone, features_only=True, pretrained=pretrained)
        chs = self.enc.feature_info.channels()  # [16, 24, 32, 96, 1280]
        self.up4 = self._up(chs[4], chs[3])
        self.up3 = self._up(chs[3], chs[2])
        self.up2 = self._up(chs[2], chs[1])
        self.up1 = self._up(chs[1], chs[0])
        self.out = nn.Conv2d(chs[0], out_ch, 1)

    def _up(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        f0, f1, f2, f3, f4 = self.enc(x)
        y = self.up4(f4); y = F.interpolate(y, size=f3.shape[-2:], mode="bilinear", align_corners=False); y = y + f3
        y = self.up3(y);  y = F.interpolate(y, size=f2.shape[-2:], mode="bilinear", align_corners=False); y = y + f2
        y = self.up2(y);  y = F.interpolate(y, size=f1.shape[-2:], mode="bilinear", align_corners=False); y = y + f1
        y = self.up1(y);  y = F.interpolate(y, size=f0.shape[-2:], mode="bilinear", align_corners=False); y = y + f0
        y = self.out(y)
        y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return y

