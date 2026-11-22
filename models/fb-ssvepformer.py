"""独立的 SSVEPformer 子网和可学习的子带融合。"""

import numpy as np
import torch
from torch import nn

from data_preprocess.features import filter_bands, make_filter_bank, transform_subjects, window_features
from .ssvepformer import SSVEPformer, model_config


class FBSSVEPformer(nn.Module):
    def __init__(self, cfg, num_bands):
        super().__init__()
        self.subnetworks = nn.ModuleList(SSVEPformer(cfg) for _ in range(num_bands))
        self.fusion = nn.Conv1d(num_bands, 1, kernel_size=1, bias=False)
        nn.init.normal_(self.fusion.weight, mean=0, std=cfg.init_std)

    def forward(self, x):
        # 每个子网保有独立权重，融合层在各类别上使用同一组子带权重。
        scores = torch.stack([network(x[:, band]) for band, network in enumerate(self.subnetworks)], dim=1)
        return self.fusion(scores).squeeze(1)


def prepare_inputs(archive, count, settings):
    fs = archive["sampling_rate_hz"].item()
    filters = make_filter_bank(
        fs, settings["subbands_hz"], settings["lower_transition_hz"], settings["upper_stop_hz"],
        settings["passband_ripple_db"], settings["stopband_attenuation_db"],
    )

    def make_features(trials, start=0):
        features = [window_features(band[..., start : start + count], fs, settings)
                    for band in filter_bands(trials, filters)]
        return np.stack(features, axis=1)

    if archive.get("preprocess_id", 1) == 2:
        start = int(archive["window_start_sample"])

        def subject_features(subject):
            return make_features(archive[f"raw_trials_s{subject}"], start)

        return (transform_subjects(archive["subject_ids"], subject_features),)
    return (transform_subjects(archive[f"raw_{count}"], make_features),)


def build_model(shapes, num_classes, settings):
    num_bands, channels, features = shapes[0]
    return FBSSVEPformer(model_config((channels, features), num_classes, settings), num_bands)


def training_plan(settings):
    stages = [(f"subband_{band + 1}", settings["pretrain_epochs"], band)
              for band in range(len(settings["subbands_hz"]))]
    return stages + [("joint", settings["joint_epochs"], None)]
