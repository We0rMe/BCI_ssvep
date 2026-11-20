"""SpikeTFNet：频域卷积编码、SpikingJelly 时域分支及特征门控。"""

from types import SimpleNamespace

import torch
from torch import nn
from spikingjelly.activation_based import functional, neuron, surrogate

from data_preprocess.features import transform_subjects, window_features


def projection(input_dim, output_dim, dropout):
    return nn.Sequential(nn.Linear(input_dim, output_dim),
                         nn.LayerNorm(output_dim),
                         nn.GELU(),
                         nn.Dropout(dropout))


def lif_node(cfg):
    return neuron.LIFNode(tau=cfg.lif_tau, decay_input=cfg.lif_decay_input,
                          v_threshold=cfg.lif_threshold, v_reset=cfg.lif_reset,
                          surrogate_function=surrogate.Sigmoid(alpha=cfg.surrogate_alpha),
                          detach_reset=cfg.detach_reset, step_mode="s", backend="torch")

class SpectralBlock(nn.Module):
    def __init__(self, channels, features, cfg):
        super().__init__()
        self.pre_conv = nn.LayerNorm(features)
        self.conv = nn.Sequential(nn.Conv1d(channels, channels, cfg.spectral_kernel, padding=cfg.spectral_kernel // 2),
                                  nn.LayerNorm(features),
                                  nn.GELU(),
                                  nn.Dropout(cfg.dropout))
        self.pre_ff = nn.LayerNorm(features)
        # 原 SSVEPformer 的 channel MLP 在每个通道的频谱长度轴上变换。
        self.ff = nn.Sequential(nn.Linear(features, features),
                                nn.GELU(),
                                nn.Dropout(cfg.dropout))

    def forward(self, x):
        x = x + self.conv(self.pre_conv(x))
        return x + self.ff(self.pre_ff(x))


class SpectralBranch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        channels, features = 2 * cfg.num_channels, cfg.spectral_features
        self.combine = nn.Sequential(nn.Conv1d(cfg.num_channels, channels, 1),
                                    nn.LayerNorm(features),
                                    nn.GELU(),
                                    nn.Dropout(cfg.dropout))
        self.blocks = nn.Sequential(*(SpectralBlock(channels, features, cfg) for _ in range(cfg.spectral_depth)))
        flat, hidden = channels * features, cfg.num_classes * cfg.spectral_head_multiplier
        self.classifier = nn.Sequential(nn.Dropout(cfg.dropout),
                                        nn.Linear(flat, hidden),
                                        nn.LayerNorm(hidden),
                                        nn.GELU(),
                                        nn.Dropout(cfg.dropout),
                                        nn.Linear(hidden, cfg.num_classes))
        self.embedding = projection(flat, cfg.embedding_dim, cfg.dropout)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.normal_(module.weight, std=cfg.spectral_init_std)

    def forward(self, spectrum):
        flat = self.blocks(self.combine(spectrum)).flatten(1)
        return self.embedding(flat), self.classifier(flat)


class SplitBiasConv1d(nn.Conv1d):
    """保留原 weight/bias 参数，避开旧 CUDA 膨胀实现的逐通道偏置填充。"""

    def forward(self, x):
        output = self._conv_forward(x, self.weight, None)
        if self.bias is not None:
            shape = (1, -1, 1) if output.ndim == 3 else (-1, 1)
            output = output + self.bias.view(shape)
        return output


class SpikingResidualBlock(nn.Module):
    def __init__(self, cfg, dilation):
        super().__init__()
        width = cfg.temporal_width
        # 后两个空洞卷积层使用 split_bias 以避开旧 CUDA 的非确定性
        conv = SplitBiasConv1d if cfg.t2_conv_execution == "split_bias" and dilation > 1 else nn.Conv1d
        self.dilated = nn.Sequential(conv(width, width, cfg.residual_kernel, padding=dilation * (cfg.residual_kernel // 2), dilation=dilation),
                                     nn.BatchNorm1d(width),
                                     lif_node(cfg))
        self.pointwise = nn.Sequential(nn.Conv1d(width, width, 1),
                                       nn.BatchNorm1d(width),
                                       lif_node(cfg))
        self.dropout = nn.Dropout(cfg.temporal_backbone_dropout)

    def forward(self, x):
        return x + self.dropout(self.pointwise(self.dilated(x)))


class SpikeAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads, self.scale = cfg.attention_heads, cfg.attention_scale
        width = cfg.temporal_width
        self.length = cfg.num_samples
        head_width = width // self.heads

        # 各头独立实现 Z @ W_Q/K/V；BN 为每个时间位置汇总批次和该头的特征。
        self.qkv = nn.ModuleList([
            nn.ModuleList([nn.Sequential(nn.Linear(width, head_width, bias=False),
                                         nn.BatchNorm1d(self.length),
                                         lif_node(cfg)) for _ in range(3)]) for _ in range(self.heads)
        ])
        self.attended = nn.ModuleList([
            nn.Sequential(nn.BatchNorm1d(self.length),
                          lif_node(cfg)) for _ in range(self.heads)
        ])
        self.output = nn.Sequential(nn.Linear(width, width, bias=False),
                                    nn.BatchNorm1d(self.length),
                                    lif_node(cfg))

    def forward(self, x):


        x = x.transpose(-1, -2)
        heads = []
        for projections, attended in zip(self.qkv, self.attended):

            q, k, v = [layer(x) for layer in projections]

            current = self.scale * ((q @ k.transpose(-1, -2)) @ v)

            # 各头独立 BN + SN，不能把不同头合入批次共享 BN 或膜电位。
            heads.append(attended(current))

        current = torch.cat(heads, dim=-1)
        return x + self.output(current)


class TemporalBranch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.steps = cfg.simulation_steps
        self.readout_bins = cfg.temporal_readout_bins
        self.current = nn.Sequential(nn.Conv1d(cfg.num_channels, cfg.temporal_width, cfg.stem_kernel, padding=cfg.stem_kernel // 2),
                                     nn.BatchNorm1d(cfg.temporal_width))
        self.encoder = lif_node(cfg)
        self.blocks = nn.Sequential(*(SpikingResidualBlock(cfg, dilation) for dilation in cfg.dilations))
        self.attention = SpikeAttention(cfg)
        self.embedding = projection(cfg.temporal_width * self.readout_bins, cfg.embedding_dim, cfg.dropout)
        self.classifier = nn.Linear(cfg.embedding_dim, cfg.num_classes)

    def pool_features(self, features):
        if self.readout_bins == 1:
            return features.mean(dim=-1)  # 原路径保留，兼容历史模型及随机初始化。
        # 使用相同的 floor/ceil 分段，避开 CUDA adaptive pooling 的非确定性反向。
        length, bins = features.shape[-1], self.readout_bins
        segments = [features[..., i * length // bins: ((i + 1) * length + bins - 1) // bins].mean(dim=-1) for i in range(bins)]
        return torch.stack(segments, dim=-1).flatten(1)

    def forward(self, time):
        current = self.current(time)  # 恒定电流只计算一次，stem BN 也只更新一次
        embeddings, logits = [], []
        for _ in range(self.steps):

            spikes = self.encoder(current)

            features = self.attention(self.blocks(spikes))

            features = features.transpose(-1, -2)

            embedding = self.embedding(self.pool_features(features))

            embeddings.append(embedding)

            logits.append(self.classifier(embedding))

        return torch.stack(embeddings).mean(0), torch.stack(logits).mean(0)


class GatedFusion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.spectral = projection(cfg.embedding_dim, cfg.fusion_dim, cfg.dropout)
        self.temporal = projection(cfg.embedding_dim, cfg.fusion_dim, cfg.dropout)
        self.gate = nn.Sequential(nn.LayerNorm(4 * cfg.fusion_dim), nn.Linear(4 * cfg.fusion_dim, cfg.gate_hidden_dim),
                                  nn.GELU(), nn.Linear(cfg.gate_hidden_dim, cfg.fusion_dim), nn.Sigmoid())
        self.classifier = nn.Linear(cfg.fusion_dim, cfg.num_classes)

    @staticmethod
    def blend(spectral, temporal, gate):
        return (1 - gate) * spectral + gate * temporal

    def forward(self, spectral, temporal):
        spectral, temporal = self.spectral(spectral), self.temporal(temporal)
        relations = torch.cat((spectral, temporal, spectral * temporal, (spectral - temporal).abs()), dim=-1)
        gate = self.gate(relations)
        fused = self.blend(spectral, temporal, gate)
        return self.classifier(fused), gate, fused


class SpikeTFNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.spectral = SpectralBranch(cfg)
        self.temporal = TemporalBranch(cfg)
        self.fusion = GatedFusion(cfg)

    def forward(self, time, spectrum):
        functional.reset_net(self.temporal)
        try:
            hf, of = self.spectral(spectrum)
            ht, ot = self.temporal(time)
            logits, gate, embedding = self.fusion(hf, ht)
            return {"logits": logits, "spectral_logits": of, "temporal_logits": ot,
                    "gate": gate, "embedding": embedding}
        finally:
            # 清理状态不切断输出的计算图；不同 trial/batch 不共享膜电位。
            functional.reset_net(self.temporal)


def joint_loss(outputs, labels, settings):
    ce = nn.functional.cross_entropy
    return (ce(outputs["logits"], labels) + settings["lambda_spectral"] * ce(outputs["spectral_logits"], labels)
            + settings["lambda_temporal"] * ce(outputs["temporal_logits"], labels))



def prepare_inputs(archive, count, settings):
    eeg = archive[f"eeg_{count}"]
    fs = archive["sampling_rate_hz"].item()
    spectrum = transform_subjects(eeg, lambda trials: window_features(trials, fs, settings))
    return eeg, spectrum


def build_model(shapes, num_classes, settings):
    (channels, samples), (spectral_channels, spectral_features) = shapes
    if (channels != spectral_channels or settings["temporal_readout_bins"] > samples
            or settings["temporal_width"] % settings["attention_heads"]):
        raise ValueError("检查 SpikeTFNet 的频谱布局、时域读出段数和注意力头数")
    cfg = SimpleNamespace(
        num_channels=channels, num_samples=samples, num_classes=num_classes,
        spectral_features=spectral_features, **settings,
    )
    return SpikeTFNet(cfg)
