"""逐试次 RMS 归一化复频谱的线性分类器。"""

import numpy as np
from torch import nn

from data_preprocess.features import transform_subjects, window_features


def prepare_inputs(archive, count, settings):
    eeg = archive[f"eeg_{count}"]
    fs = archive["sampling_rate_hz"].item()

    def trial_spectra(trials):
        # 每个试次的电极和采样点共用一个 RMS，保留电极间相对幅值。
        rms = np.sqrt(np.square(trials, dtype=np.float64).mean(axis=(-2, -1), keepdims=True))
        normalized = (trials / np.maximum(rms, settings["rms_floor"])).astype(np.float32)
        return window_features(normalized, fs, settings)

    return (transform_subjects(eeg, trial_spectra),)


def build_model(shapes, num_classes, settings):
    (channels, features), = shapes
    return nn.Sequential(nn.Flatten(), nn.Linear(channels * features, num_classes))
