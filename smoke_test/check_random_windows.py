"""真实整试次上的边界、标签、滤波顺序、断点恢复和留出评估。"""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
from scipy.signal import sosfiltfilt

import evaluate
import train
from configs import config
from data_preprocess import config as data
from data_preprocess.random_windows import RandomWindows, sampling_config
from smoke_test.check_ifuzzytl import check_resume


def small_archive():
    trials = [0, 3, 42, 119]
    with np.load(data.OUTPUT_FILES[2]) as source:
        arrays = {key: source[key] for key in ("subject_ids", "labels", "block_ids", "sampling_rate_hz",
                  "channels", "butterworth_order", "bandpass_hz", "window_start_sample", "preprocess_id")}
        arrays["subject_ids"] = arrays["subject_ids"][:3]
        for key in ("labels", "block_ids"):
            arrays[key] = arrays[key][trials]
        for s in arrays["subject_ids"]:
            key = f"raw_trials_s{s}"
            arrays[key] = source[key][trials]
        for count in (125, 250):
            for prefix in ("raw", "eeg"):
                key = f"{prefix}_{count}"
                arrays[key] = source[key][:3, trials]
    return arrays


def check_windows(arrays, count, method, name, sampling):
    module, options = train.model_module(name), train.model_config(name).SETTINGS
    windows = RandomWindows(module, arrays, count, options, sampling, method)
    split = {"test": 1, "train": [2, 3]}
    subjects, trials, starts = windows.draw(split, 42, epoch=0)
    assert set(subjects) == {1, 2} and len(starts) == 12000
    assert starts.min() == 125 and starts.max() == 1375 - count
    assert not np.array_equal(starts, windows.draw(split, 42, epoch=1)[2])
    np.testing.assert_array_equal(starts, windows.draw(split, 42, epoch=0)[2])
    subjects, trials, starts = windows.draw(split, 42)
    np.testing.assert_array_equal(trials, np.arange(4))
    assert (subjects == 0).all()
    if name == "ifuzzytl":
        # 同一批内混合不同被试、trial 和起点，含合法区间首尾。
        selected = (np.array([2, 0, 1]), np.array([1, 3, 0]), np.array([125, 200, 1375 - count]))
        eeg, labels = next(windows.batches(selected, 3))
        for i, (subject, trial, start) in enumerate(zip(*selected)):
            raw = arrays[f"raw_trials_s{subject + 1}"][trial]
            cut = slice(start, start + count)
            ref = (sosfiltfilt(windows.sos, raw[..., cut]) if method == 1
                   else sosfiltfilt(windows.sos, raw)[..., cut])
            np.testing.assert_allclose(eeg[i], ref, atol=1e-5, rtol=1e-5)
            assert labels[i] == arrays["labels"][trial]
    # 强制选取已知起点，与原来的固定窗输入逐元素比较。
    starts[:] = 160
    features = next(windows.batches((subjects, trials, starts), 4))
    fixed = dict(arrays, preprocess_id=method)
    if method == 1:
        with np.load(data.OUTPUT_FILES[1]) as archive:
            fixed[f"eeg_{count}"] = archive[f"eeg_{count}"][:3, [0, 3, 42, 119]]
    for actual, expected in zip(features[:-1], module.prepare_inputs(fixed, count, options), strict=True):
        np.testing.assert_allclose(actual.numpy(), expected[0], atol=1e-5, rtol=1e-5)
    np.testing.assert_array_equal(features[-1], arrays["labels"])


def main():
    torch.set_num_threads(2)
    output = data.PROJECT_DIR / "smoke_test/output/random_windows"
    output.mkdir(parents=True, exist_ok=True)
    arrays = small_archive()
    with patch.multiple(config, WINDOW_MODE="random", RANDOM_START_SECONDS=0.5, RANDOM_WINDOWS_PER_EPOCH=12000):
        sampling = sampling_config(config, (125, 250))
        for count in (125, 250):
            for method in (1, 2):
                for name in ("ifuzzytl", "fb-ssvepformer"):
                    check_windows(arrays, count, method, name, sampling)
        with TemporaryDirectory(dir=output) as tmp, patch.object(config, "RANDOM_WINDOWS_PER_EPOCH", 12):
            directory = Path(tmp)
            path = directory / "small.npz"
            np.savez(path, **arrays)
            check_resume(path, directory)
            predictions = directory / "resumed/ifuzzytl/artifacts/predictions.npz"
            with np.load(predictions) as result:
                saved = dict(result)
            with patch.multiple(config, DEVICE="cpu", WINDOW_MODE="fixed", RANDOM_START_SECONDS=0.64):
                evaluate.evaluate_experiment(path, directory / "resumed")
            with np.load(predictions) as result:
                for count in (125, 250):
                    assert result[f"window_start_samples_{count}"].shape == (1, 4)
                for key in saved:
                    np.testing.assert_array_equal(result[key], saved[key])
    report = "CPU 冒烟通过：两个窗长、两种滤波顺序、普通及子带输入、训练被试隔离、跨轮重采样、断点恢复和测试窗回放。\n"
    (output / "cpu.txt").write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
