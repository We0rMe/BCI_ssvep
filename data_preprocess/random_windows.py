"""从完整 trial 抽窗。训练按折和轮次重采样，测试每条 trial 保留一个可复查的起点。"""

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt

from . import config as data
from .features import filter_bands, window_features


def sampling_config(config, counts):
    if config.WINDOW_MODE == "fixed":
        return None
    if config.WINDOW_MODE != "random":
        raise ValueError("WINDOW_MODE 须为 fixed 或 random")
    fs = data.SAMPLING_RATE_HZ
    start = config.RANDOM_START_SECONDS * fs
    stop = round((data.CUE_SECONDS + data.STIMULUS_SECONDS) * fs)
    if not (start >= data.CUE_SECONDS * fs and start == round(start) and start + max(counts) <= stop):
        raise ValueError("随机窗下界须位于刺激段内、对齐采样点，并为最长窗留出足够空间")
    size = config.RANDOM_WINDOWS_PER_EPOCH
    if type(size) is not int or size < 1:
        raise ValueError("每轮随机窗数须为正整数")
    return dict(mode="random", first_sample=round(start), stop_sample=stop, train_windows=size,
                test_windows_per_trial=1, sampler_version=1)


class RandomWindows:
    def __init__(self, module, archive, count, options, sampling, method):
        self.module, self.options = module, options
        self.count, self.sampling, self.method = count, sampling, method
        self.fs = archive["sampling_rate_hz"]
        self.labels = archive["labels"]
        ids = archive["subject_ids"]
        self.indices = {int(subject): i for i, subject in enumerate(ids)}
        self.sos = butter(data.BUTTERWORTH_ORDER, data.BANDPASS_HZ, btype="bandpass", fs=self.fs.item(), output="sos")
        self.fb = hasattr(module, "full_trial_bands")
        bands = module.full_trial_bands(self.fs.item(), options) if self.fb and method == 2 else None
        for i, subject in enumerate(ids):
            raw = archive[f"raw_trials_s{subject}"]
            if bands is not None:
                signal = np.stack(list(filter_bands(raw, bands)), axis=1)
            elif method == 2:
                signal = sosfiltfilt(self.sos, raw, axis=-1).astype(np.float32)
            else:
                signal = raw
            # [被试, trial, 时间, ...] 便于一次索引整个 batch 的不同起点。
            signal = np.moveaxis(signal, -1, 1)
            if i == 0:
                self.signals = np.empty((len(ids), *signal.shape), dtype=signal.dtype)
            self.signals[i] = signal

    def draw(self, split, seed, epoch=None):
        rng = np.random.default_rng([seed, split["test"], self.count, 0 if epoch is None else epoch + 1])
        if epoch is None:
            trials = np.arange(len(self.labels))
            subjects = np.full(len(trials), self.indices[split["test"]])
        else:
            size = self.sampling["train_windows"]
            subjects = rng.choice([self.indices[s] for s in split["train"]], size)
            trials = rng.integers(len(self.labels), size=size)
        starts = rng.integers(self.sampling["first_sample"], self.sampling["stop_sample"] - self.count + 1,
                              size=len(trials))
        return subjects, trials, starts

    def features(self, windows):
        if self.fb and self.method == 2:
            return (window_features(windows, self.fs.item(), self.options),)
        batch = {"sampling_rate_hz": self.fs}
        if self.fb:
            batch[f"raw_{self.count}"] = windows[None]
        else:
            if self.method == 1:
                windows = sosfiltfilt(self.sos, windows, axis=-1)
            batch[f"eeg_{self.count}"] = windows.astype(np.float32, copy=False)[None]
        return tuple(x[0] for x in self.module.prepare_inputs(batch, self.count, self.options))

    def shapes(self):
        sample = np.zeros((1, *self.signals.shape[3:], self.count), dtype=self.signals.dtype)
        return tuple(x.shape[1:] for x in self.features(sample))

    def batches(self, selection, batch_size):
        subjects, trials, starts = selection
        offsets = np.arange(self.count)
        for begin in range(0, len(trials), batch_size):
            sl = slice(begin, begin + batch_size)
            times = starts[sl, None] + offsets
            windows = self.signals[subjects[sl, None], trials[sl, None], times]
            values = self.features(np.ascontiguousarray(np.moveaxis(windows, 1, -1)))
            yield (*[torch.from_numpy(x) for x in values], torch.from_numpy(self.labels[trials[sl]]))
