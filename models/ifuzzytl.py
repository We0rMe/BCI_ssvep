"""iFuzzyTL：用高斯规则分别对电极和时间点加权。"""

import math

import torch
from torch import nn


def prepare_inputs(archive, count, settings):
    return (archive[f"eeg_{count}"],)


class FuzzyFilter(nn.Module):
    def __init__(self, features, rules, width_scale):
        super().__init__()
        self.rules = rules
        # 一次线性计算得到所有规则的投影，各规则有独立的权重。
        self.query = nn.Linear(features, rules * features)
        self.value = nn.Linear(features, rules * features)
        self.centers = nn.Parameter(torch.randn(rules, features) / math.sqrt(features))
        width = width_scale * features ** 0.25
        self.log_width = nn.Parameter(torch.full((rules, features), math.log(width)))

    def rule_weights(self, x):
        query = self.query(x).unflatten(-1, (self.rules, x.size(-1)))
        distance = (query - self.centers) / self.log_width.exp()
        scores = -0.5 * distance.square().sum(dim=-1)
        return scores.softmax(dim=-1)

    def forward(self, x):
        weights = self.rule_weights(x)
        values = self.value(x).unflatten(-1, (self.rules, x.size(-1)))
        return (weights.unsqueeze(-1) * values).sum(dim=-2)


class IFuzzyTL(nn.Module):
    def __init__(self, channels, samples, num_classes, settings):
        super().__init__()
        self.spatial = FuzzyFilter(samples, settings["spatial_rules"], settings["spatial_width_scale"])
        self.temporal = FuzzyFilter(channels, settings["temporal_rules"], settings["temporal_width_scale"])
        self.head = nn.Sequential(
            nn.Linear(channels, channels), nn.ReLU(), nn.Dropout(settings["dropout"]),
            nn.Linear(channels, num_classes),
        )

    def forward(self, eeg):
        spatial = self.spatial(eeg)                      # [批次, 电极, 时间]
        temporal = self.temporal(spatial.transpose(1, 2))  # [批次, 时间, 电极]
        return self.head(temporal.mean(dim=1))


def build_model(shapes, num_classes, settings):
    channels, samples = shapes[0]
    return IFuzzyTL(channels, samples, num_classes, settings)
