"""MTSNet 时域、频域编码器与多尺度融合，结构依据 lanzhen19/MTSNet。"""

import torch
from torch import nn

from data_preprocess.features import transform_subjects, window_features


def prepare_inputs(archive, count, settings):
    eeg = archive[f"eeg_{count}"]
    fs = archive["sampling_rate_hz"].item()
    spectrum = transform_subjects(eeg, lambda trials: window_features(trials, fs, settings))
    return eeg, spectrum


def conv_features(in_channels, out_channels, length, kernel, dropout):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel, padding=kernel // 2),
        nn.LayerNorm(length), 
        nn.GELU(), 
        nn.Dropout(dropout),
    )


class TemporalMixer(nn.Module):
    def __init__(self, channels, length, kernels, dropout):
        super().__init__()
        self.first = conv_features(channels, channels, length, kernels[1], dropout)
        self.second = conv_features(channels, channels, length, kernels[0], dropout)

    def forward(self, x):
        first = self.first(x)
        return first + self.second(first)


class FusionMixer(nn.Module):
    def __init__(self, channels, length, kernels, dropout):
        super().__init__()
        self.branches = nn.ModuleList(
            conv_features(channels, channels, length, kernel, dropout) for kernel in kernels
        )

    def forward(self, x):
        return sum(branch(x) for branch in self.branches)


class ConvformerBlock(nn.Module):
    def __init__(self, length, mixer, dropout):
        super().__init__()
        self.normalization = nn.LayerNorm(length)
        self.mixer = mixer
        self.mlp = nn.Sequential(
            nn.LayerNorm(length), nn.Linear(length, 2 * length), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * length, length), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.mixer(self.normalization(x))
        return x + self.mlp(x)


class MTSNet(nn.Module):
    def __init__(self, shapes, num_classes, settings):
        super().__init__()
        (channels, samples), (spectral_channels, features) = shapes
        if channels != spectral_channels:
            raise ValueError("MTSNet 时域与频域输入须使用相同通道")
        width = 2 * channels
        length = samples + features
        dropout = settings["dropout"]
        kernels = settings["kernel_sizes"]
        if len(kernels) != 3 or any(k < 1 or k % 2 == 0 for k in kernels):
            raise ValueError("MTSNet 使用三个正奇数卷积核以保持特征长度")

        time_embedding = conv_features(channels, width, samples, 1, dropout)
        spectral_embedding = conv_features(channels, width, features, 1, dropout)
        time_blocks = [
            ConvformerBlock(samples, TemporalMixer(width, samples, kernels, dropout), dropout)
            for _ in range(settings["branch_depth"])
        ]
        spectral_blocks = [
            ConvformerBlock(features, conv_features(width, width, features, kernels[0], dropout), dropout)
            for _ in range(settings["branch_depth"])
        ]
        self.temporal = nn.Sequential(time_embedding, *time_blocks)
        self.spectral = nn.Sequential(spectral_embedding, *spectral_blocks)
        self.fusion = nn.Sequential(*(
            ConvformerBlock(length, FusionMixer(width, length, kernels, dropout), dropout)
            for _ in range(settings["fusion_depth"])
        ))
        hidden = 6 * num_classes
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(width * length, hidden),
            nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.5), nn.Linear(hidden, num_classes),
        )
        # 与作者源码按相同顺序抽取初始权重。
        for part in (time_embedding, spectral_embedding, *time_blocks, *spectral_blocks, self.fusion, self.classifier):
            for layer in part.modules():
                if isinstance(layer, (nn.Conv1d, nn.Linear)):
                    nn.init.normal_(layer.weight, std=0.01)

    def forward(self, eeg, spectrum):
        # 先做两路投影，再依次编码，使各层 Dropout 的调用顺序与作者一致。
        eeg = self.temporal[0](eeg)
        spectrum = self.spectral[0](spectrum)
        for index in range(1, len(self.temporal)):
            eeg = self.temporal[index](eeg)
        for index in range(1, len(self.spectral)):
            spectrum = self.spectral[index](spectrum)
        features = torch.cat((eeg, spectrum), dim=-1)
        return self.classifier(self.fusion(features))


def build_model(shapes, num_classes, settings):
    return MTSNet(shapes, num_classes, settings)
