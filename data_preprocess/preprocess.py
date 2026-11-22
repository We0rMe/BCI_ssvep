import math
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import scipy
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt

from . import config


def window_samples(seconds_list):
    counts = []
    limit = config.STIMULUS_SECONDS - config.VISUAL_LATENCY_SECONDS
    for seconds in seconds_list:
        samples = seconds * config.SAMPLING_RATE_HZ
        count = round(samples)
        if not 0 < seconds <= limit or not math.isclose(samples, count, abs_tol=1e-8, rel_tol=0):
            raise ValueError(f"窗长 {seconds:g} s 须在 (0, {limit:g}] s 内，且对应整数个采样点")
        counts.append(count)
    if not counts or len(counts) != len(set(counts)):
        raise ValueError("窗长至少配置一个，且各窗长不能重复")
    return counts


def start_sample():
    return round((config.CUE_SECONDS + config.VISUAL_LATENCY_SECONDS) * config.SAMPLING_RATE_HZ)


def channel_indices(location_file):
    channels = {}
    for line in location_file.read_text(encoding="utf-8-sig").splitlines():
        fields = line.split()
        if fields:
            channels[fields[-1].casefold()] = int(fields[0]) - 1
    return [channels[name.casefold()] for name in config.CHANNELS]


def crop_trials(raw, channels, start, count):
    # 区块优先，区块内保持 MAT 目标轴的原始顺序。
    window = raw[channels, start : start + count, :, :]
    return window.transpose(3, 2, 0, 1).reshape(-1, len(channels), count)


def save_archive(output_file, arrays):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    file = NamedTemporaryFile(dir=output_file.parent, suffix=".npz", delete=False)
    temporary = Path(file.name)
    try:
        with file:
            np.savez(file, **arrays)
        temporary.replace(output_file)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_benchmark(data_dir, output_file, subject_ids, windows, method=1):
    if method not in config.OUTPUT_FILES:
        raise ValueError(f"未知预处理编号 {method}")
    counts = window_samples(windows)
    start = start_sample()
    channels = channel_indices(data_dir / "64-channels.loc")
    targets = loadmat(data_dir / "Freq_Phase.mat", variable_names=("freqs", "phases"))
    frequencies = targets["freqs"].ravel()
    phases = targets["phases"].ravel()
    if frequencies.size != config.NUM_CLASSES or phases.size != config.NUM_CLASSES:
        raise ValueError("频率和相位数量须与 MAT 目标轴一致")

    sos = butter(config.BUTTERWORTH_ORDER, config.BANDPASS_HZ, btype="bandpass",
                 fs=config.SAMPLING_RATE_HZ, output="sos")
    trials = config.NUM_CLASSES * config.NUM_BLOCKS
    expected = (config.NUM_CHANNELS, config.TRIAL_SAMPLES, config.NUM_CLASSES, config.NUM_BLOCKS)
    arrays = {
        "subject_ids": np.asarray(subject_ids, dtype=np.int64),
        "labels": np.tile(np.arange(config.NUM_CLASSES, dtype=np.int64), config.NUM_BLOCKS),
        "block_ids": np.repeat(np.arange(1, config.NUM_BLOCKS + 1), config.NUM_CLASSES),
        "frequencies_hz": frequencies,
        "phases_rad": phases,
        "channels": np.asarray(config.CHANNELS),
        "sampling_rate_hz": config.SAMPLING_RATE_HZ,
        "window_start_sample": start,
        "window_samples": np.asarray(counts),
        "bandpass_hz": np.asarray(config.BANDPASS_HZ),
        "butterworth_order": config.BUTTERWORTH_ORDER,
        "scipy_version": scipy.__version__,
    }
    for count in counts:
        shape = (len(subject_ids), trials, len(channels), count)
        arrays[f"raw_{count}"] = np.empty(shape, dtype=np.float64)
        arrays[f"eeg_{count}"] = np.empty(shape, dtype=np.float32)

    if method == 2:
        arrays["preprocess_id"] = method
    for index, subject_id in enumerate(subject_ids):
        source = data_dir / f"S{subject_id}.mat"
        raw = loadmat(source, variable_names=("data",))["data"]
        if raw.shape != expected:
            raise ValueError(f"{source.name} 的形状为 {raw.shape}，预期为 {expected}")
        if method == 2:
            full_trials = crop_trials(raw, channels, 0, config.TRIAL_SAMPLES)
            # FB 的子带取决于模型配置，保留原始试次供其逐被试滤波。
            arrays[f"raw_trials_s{subject_id}"] = full_trials
            filtered = sosfiltfilt(sos, full_trials, axis=-1)
        for count in counts:
            trials_in_window = crop_trials(raw, channels, start, count)
            arrays[f"raw_{count}"][index] = trials_in_window
            if method == 1:
                arrays[f"eeg_{count}"][index] = sosfiltfilt(sos, trials_in_window, axis=-1)
            else:
                arrays[f"eeg_{count}"][index] = filtered[..., start : start + count]
        print(f"[{index + 1}/{len(subject_ids)}] 已处理 S{subject_id}", flush=True)

    save_archive(output_file, arrays)
    print(f"预处理文件已保存到 {output_file}", flush=True)


def main():
    for method, path in config.OUTPUT_FILES.items():
        prepare_benchmark(config.BENCHMARK_DIR, path, range(1, config.NUM_SUBJECTS + 1),
                          config.WINDOW_SECONDS, method)


if __name__ == "__main__":
    main()
