"""SSVEPformer 的通道组合、残差编码器和分类头。"""

from types import SimpleNamespace

from torch import nn

from data_preprocess.features import transform_subjects, window_features


class ChannelLayerNorm(nn.LayerNorm):
    """输入为 [batch, channels, features]，在每个频谱位置沿通道归一化。"""

    def forward(self, x):
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


def backbone_norm(channels, features, cfg):
    # 历史运行清单没有此字段，恢复权重时沿用其频谱轴 LayerNorm。
    if getattr(cfg, "channel_layer_norm", False):
        return ChannelLayerNorm(channels)
    return nn.LayerNorm(features)


class Encoder(nn.Module):
    def __init__(self, channels, features, cfg):
        super().__init__()
        self.cnn = nn.Sequential(
            backbone_norm(channels, features, cfg),
            nn.Conv1d(channels, channels, cfg.kernel_size, padding=cfg.kernel_size // 2),
            backbone_norm(channels, features, cfg),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.mlp = nn.Sequential(
            backbone_norm(channels, features, cfg), nn.Linear(features, features), nn.GELU(), nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        x = x + self.cnn(x)
        return x + self.mlp(x)


class SSVEPformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        channels = cfg.input_channels * cfg.spatial_multiplier
        features = cfg.features
        hidden = cfg.head_multiplier * cfg.classes
        self.channel_combination = nn.Sequential(
            nn.Conv1d(cfg.input_channels, channels, 1), backbone_norm(channels, features, cfg),
            nn.GELU(), nn.Dropout(cfg.dropout),
        )
        self.encoder = nn.Sequential(*(Encoder(channels, features, cfg) for _ in range(cfg.encoder_blocks)))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(cfg.dropout), nn.Linear(channels * features, hidden),
            nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(hidden, cfg.classes),
        )
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.normal_(module.weight, mean=0, std=cfg.init_std)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.classifier(self.encoder(self.channel_combination(x)))


def model_config(input_shape, num_classes, settings):
    channels, features = input_shape
    return SimpleNamespace(input_channels=channels, features=features, classes=num_classes, **settings)


def prepare_inputs(archive, count, settings):
    eeg = archive[f"eeg_{count}"]
    if settings["input_type"] == "time":
        return (eeg,)
    fs = archive["sampling_rate_hz"].item()
    features = transform_subjects(eeg, lambda trials: window_features(trials, fs, settings))
    return (features,)


def build_model(shapes, num_classes, settings):
    return SSVEPformer(model_config(shapes[0], num_classes, settings))
