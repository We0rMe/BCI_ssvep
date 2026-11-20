import math

import numpy as np
from scipy.signal import cheb1ord, cheby1, sosfiltfilt


def window_features(windows, sample_rate, settings):
    layout = settings["input_type"]
    if layout == "time":
        return windows.astype(np.float32, copy=False)

    resolution = settings["fft_resolution"]
    nfft = round(sample_rate / resolution)
    low, high = settings["spectrum_hz"]
    first, last = (round(value / resolution) for value in (low, high))
    if (windows.shape[-1] > nfft or not 0 <= low < high <= sample_rate / 2
            or not math.isclose(sample_rate / nfft, resolution)
            or not math.isclose(first * resolution, low) or not math.isclose(last * resolution, high)):
        raise ValueError("FFT 长度须覆盖数据窗，频段端点须落在有效频点上")

    spectrum = np.fft.rfft(windows, n=nfft, axis=-1)[..., first : last + int(settings["include_upper_bin"])]
    spectrum *= settings["fft_scale"] / windows.shape[-1]
    if layout == "amplitude":
        return np.abs(spectrum).astype(np.float32)
    axis = {"complex": -1, "complex_channels": -2}[layout]
    return np.concatenate((spectrum.real, spectrum.imag), axis=axis).astype(np.float32)


def make_filter_bank(sample_rate, passbands, lower_transition, upper_stop, ripple, attenuation):
    filters = []
    for low, high in passbands:
        stopband = (low - lower_transition, upper_stop)
        order, cutoff = cheb1ord((low, high), stopband, ripple, attenuation, fs=sample_rate)
        filters.append(cheby1(order, ripple, cutoff, btype="bandpass", fs=sample_rate, output="sos"))
    return filters


def filter_bands(windows, filters):
    """逐个产出从同一分类窗提取的子带。"""
    for sos in filters:
        yield sosfiltfilt(sos, windows, axis=-1)


def transform_subjects(windows, transform):
    # 逐被试计算，限制 FFT 和滤波器组的临时内存。
    first = transform(windows[0])
    features = np.empty((len(windows), *first.shape), dtype=np.float32)
    features[0] = first
    for index in range(1, len(windows)):
        features[index] = transform(windows[index])
    return features
