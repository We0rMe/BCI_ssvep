"""跨被试 AWKNet：三级时间小波、多尺度时空卷积和 KAN 读出。"""

import math
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from models.ssvepformer_eKAN import KANLinear


def prepare_inputs(archive, count, settings):
    return (archive[f"eeg_{count}"],)


class Wavelet(nn.Module):
    def __init__(self):
        super().__init__()
        # db4 低通滤波器，按 PyTorch 互相关的方向排列。
        self.low = nn.Parameter(torch.tensor([
            0.2303778133, 0.7148465706, 0.6308807679, -0.0279837694,
            -0.1870348117, 0.0308413818, 0.0328830117, -0.0105974018,
        ]))
        self.register_buffer("sign", torch.tensor([1., -1.] * 4))

    def forward(self, eeg):
        batch, channels, samples = eeg.shape
        low = eeg.reshape(batch * channels, 1, samples)
        filters = torch.stack((self.low, self.low.flip(0) * self.sign)).unsqueeze(1)
        details = []
        for _ in range(3):
            # 周期延拓；奇数长度先补最后一个点，使每级长度为 ceil(T/2)。
            if low.size(-1) % 2:
                low = torch.cat((low, low[..., -1:]), dim=-1)
            padded = torch.cat((low[..., -3:], low, low[..., :3]), dim=-1)
            bands = F.conv1d(padded, filters, stride=2)
            low = bands[:, :1]
            details.append(bands[:, 1].reshape(batch, channels, -1))
        return low.reshape(batch, channels, -1), details[2], details[1]

    def penalty(self):
        # 单位能量、低通直流增益和非零偶数位移的正交性。
        low = self.low
        shifts = torch.stack([(low[:-s] * low[s:]).sum() for s in range(2, low.numel(), 2)])
        return (low.square().sum() - 1).square() + (low.sum() - math.sqrt(2)).square() + shifts.square().sum()


class ScaleEncoder(nn.Module):
    def __init__(self, bands, channels, cfg):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(bands, bands, (1, k), padding=(0, k // 2), groups=bands, bias=False),
                nn.Conv2d(bands, cfg.filters_per_kernel, (channels, 1), bias=False),
            )
            for k in cfg.kernel_sizes
        ])
        width = len(cfg.kernel_sizes) * cfg.filters_per_kernel
        self.output = nn.Sequential(nn.BatchNorm2d(width), nn.ELU(), nn.AvgPool2d((1, cfg.pool_size)))

    def forward(self, x):
        return self.output(torch.cat([branch(x) for branch in self.branches], dim=1)).flatten(1)


class AWKNet(nn.Module):
    def __init__(self, shape, num_classes, cfg):
        super().__init__()
        channels, samples = shape
        self.wavelet = Wavelet()
        self.wavelet_lr = cfg.wavelet_lr
        self.coarse = ScaleEncoder(2, channels, cfg)
        self.detail = ScaleEncoder(1, channels, cfg) if cfg.use_d2 else None
        positions = math.ceil(samples / 8) // cfg.pool_size
        if cfg.use_d2:
            positions += math.ceil(samples / 4) // cfg.pool_size
        features = positions * len(cfg.kernel_sizes) * cfg.filters_per_kernel
        self.head = nn.Sequential(nn.LayerNorm(features), KANLinear(features, num_classes, cfg))

    def optimizer_groups(self):
        backbone = [p for p in self.parameters() if p is not self.wavelet.low]
        return [{"params": backbone},
                {"params": [self.wavelet.low], "lr": self.wavelet_lr, "weight_decay": 0.0}]

    def forward(self, eeg):
        a3, d3, d2 = self.wavelet(eeg)
        features = self.coarse(torch.stack((a3, d3), dim=1))
        if self.detail is not None:
            features = torch.cat((features, self.detail(d2.unsqueeze(1))), dim=1)
        return {"logits": self.head(features), "wavelet_penalty": self.wavelet.penalty()}


def joint_loss(output, targets, settings):
    return F.cross_entropy(output["logits"], targets) + settings["wavelet_penalty"] * output["wavelet_penalty"].mean()


def build_model(shapes, num_classes, settings):
    return AWKNet(shapes[0], num_classes, SimpleNamespace(**settings))
