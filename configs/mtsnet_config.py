SETTINGS = {
    # 同一滤波试次的复频谱，先实部、后虚部。
    "input_type": "complex",
    "fft_resolution": 0.2,
    "spectrum_hz": (8.0, 64.0),
    "include_upper_bin": False,
    "fft_scale": 2.0,
    "branch_depth": 2,
    "fusion_depth": 2,
    "kernel_sizes": (31, 15, 7),
    "dropout": 0.5,
}
