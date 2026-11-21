"""SsvepGPT：借鉴 SleepGPT 的统一时频编码，监督学习单试次 SSVEP 类别。"""

import torch
from torch import nn
from torch.nn import functional as F

from data_preprocess.features import transform_subjects, window_features


def prepare_inputs(archive, count, settings):
    eeg = archive[f"eeg_{count}"]
    fs = archive["sampling_rate_hz"].item()
    spectrum = transform_subjects(eeg, lambda trials: window_features(trials, fs, settings))
    return eeg, spectrum


def feed_forward(width, hidden, dropout):
    return nn.Sequential(
        nn.LayerNorm(width), nn.Linear(width, hidden), nn.GELU(),
        nn.Dropout(dropout), nn.Linear(hidden, width), nn.Dropout(dropout),
    )


class TimeFrequencyBlock(nn.Module):
    def __init__(self, settings, shared_ffn):
        super().__init__()
        width = settings["width"]
        hidden = width * settings["ffn_ratio"]
        dropout = settings["dropout"]
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, settings["attention_heads"], batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.shared_ffn = shared_ffn
        if shared_ffn:
            self.joint_ffn = feed_forward(width, hidden, dropout)
        else:
            self.time_ffn = feed_forward(width, hidden, dropout)
            self.spectrum_ffn = feed_forward(width, hidden, dropout)

    def forward(self, tokens, time_count):
        x = self.norm(tokens)
        attention, _ = self.attention(x, x, x, need_weights=False)
        tokens = tokens + self.dropout(attention)
        if self.shared_ffn:
            return tokens + self.joint_ffn(tokens)
        time, spectrum = tokens[:, :time_count], tokens[:, time_count:]
        return torch.cat((time + self.time_ffn(time), spectrum + self.spectrum_ffn(spectrum)), dim=1)


class SsvepGPT(nn.Module):
    def __init__(self, shapes, num_classes, settings):
        super().__init__()
        (channels, samples), (spectral_channels, features) = shapes
        if channels != spectral_channels:
            raise ValueError("时域和频域输入的电极数量须一致")
        width = settings["width"]
        self.patch_samples = settings["patch_samples"]
        self.padding = -samples % self.patch_samples
        patches = (samples + self.padding) // self.patch_samples
        self.time_projection = nn.Linear(self.patch_samples, width)
        bins = features // 2  # 输入先放实部，再放相同频点的虚部
        self.spectrum_bands = settings.get("spectrum_bands", 1)  # 旧实验使用整谱投影
        if features % 2 or not 1 <= self.spectrum_bands <= bins:
            raise ValueError("复频谱须含成对的实虚部，频带数须介于 1 和频点数之间")
        self.band_bins = (bins + self.spectrum_bands - 1) // self.spectrum_bands
        self.spectrum_projection = nn.Linear(2 * self.band_bins, width)
        if self.spectrum_bands > 1:
            self.band_embed = nn.Parameter(torch.empty(self.spectrum_bands, width))
            nn.init.normal_(self.band_embed, std=0.02)
        self.channel_embed = nn.Parameter(torch.empty(channels, width))
        self.position_embed = nn.Parameter(torch.empty(patches, width))
        self.domain_embed = nn.Parameter(torch.empty(2, width))
        for embedding in (self.channel_embed, self.position_embed, self.domain_embed):
            nn.init.normal_(embedding, std=0.02)

        self.blocks = nn.ModuleList()
        for _ in range(settings["domain_blocks"]):
            self.blocks.append(TimeFrequencyBlock(settings, shared_ffn=False))
        for _ in range(settings["shared_blocks"]):
            self.blocks.append(TimeFrequencyBlock(settings, shared_ffn=True))
        self.norm = nn.LayerNorm(width)
        self.classifier = nn.Linear(2 * width, num_classes)

    def spectrum_tokens(self, spectrum):
        if self.spectrum_bands == 1:
            return self.spectrum_projection(spectrum) + self.channel_embed + self.domain_embed[1]

        # 只切分频率轴。较短频带的实部、虚部分别在末尾补零。
        paired = spectrum.unflatten(-1, (2, -1))
        bands = torch.tensor_split(paired, self.spectrum_bands, dim=-1)
        padded = [F.pad(band, (0, self.band_bins - band.size(-1))) for band in bands]
        features = torch.stack(padded, dim=2).flatten(-2)  # [批次, 电极, 频带, 实部和虚部]
        tokens = self.spectrum_projection(features) + self.channel_embed[:, None] + self.band_embed
        return (tokens + self.domain_embed[1]).flatten(1, 2)

    def forward(self, eeg, spectrum):
        if self.padding:
            eeg = F.pad(eeg, (0, self.padding))
        patches = eeg.unfold(-1, self.patch_samples, self.patch_samples)
        time = self.time_projection(patches) + self.channel_embed[:, None] + self.position_embed + self.domain_embed[0]
        time = time.flatten(1, 2)  # 电极优先，每个电极内按时间排列
        spectrum = self.spectrum_tokens(spectrum)
        tokens = torch.cat((time, spectrum), dim=1)
        time_count = time.size(1)
        for block in self.blocks:
            tokens = block(tokens, time_count)
        tokens = self.norm(tokens)
        # 分域平均，两个域在读出时各占一半维度。
        pooled = torch.cat((tokens[:, :time_count].mean(1), tokens[:, time_count:].mean(1)), dim=-1)
        return self.classifier(pooled)


def build_model(shapes, num_classes, settings):
    return SsvepGPT(shapes, num_classes, settings)
