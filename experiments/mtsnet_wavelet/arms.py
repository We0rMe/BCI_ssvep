"""给 MTSNet 的时域输入加一条小波残差，复频谱分支仍使用原始输入。"""

import torch
from torch import nn
from torch.nn import functional as F

from models import mtsnet
from models.awknet import Wavelet


class WaveletMTSNet(mtsnet.MTSNet):
    def __init__(self, shapes, classes, settings):
        super().__init__(shapes, classes, settings)
        self.wavelet = Wavelet()
        if settings["arm"] == "fixed":
            # 常量也进入 checkpoint，并跟随模型移动到相应设备。
            low = self.wavelet.low.detach()
            del self.wavelet.low
            self.wavelet.register_buffer("low", low)
        self.gains = nn.Parameter(torch.zeros(3))

    def forward(self, eeg, spectrum):
        # 不下采样，通过扩张滤波得到同长度的 A3、D3、D2。
        batch, channels, samples = eeg.shape
        low = eeg.reshape(-1, 1, samples)
        taps = self.wavelet.low
        filters = torch.stack((taps, taps.flip(0) * self.wavelet.sign)).unsqueeze(1)
        details = []
        for step in (1, 2, 4):
            padded = torch.cat((low[..., -3 * step:], low, low[..., :4 * step]), dim=-1)
            bands = F.conv1d(padded, filters, dilation=step)
            low = bands[:, :1]
            details.append(bands[:, 1:2])
        residual = self.gains[0] * low + self.gains[1] * details[2] + self.gains[2] * details[1]
        logits = super().forward(eeg + residual.reshape(batch, channels, samples), spectrum)
        return {"logits": logits, "wavelet_penalty": self.wavelet.penalty()}


def build_model(shapes, num_classes, settings):
    return WaveletMTSNet(shapes, num_classes, settings)


def joint_loss(output, labels, settings):
    return F.cross_entropy(output["logits"], labels) + settings["wavelet_penalty"] * output["wavelet_penalty"]
