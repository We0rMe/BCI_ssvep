SETTINGS = {
    "input_type": "complex",  # 每个电极的整窗频谱，先实部、后虚部
    "fft_resolution": 0.2,
    "spectrum_hz": (8.0, 64.0),
    "include_upper_bin": True,
    "fft_scale": 1.0,
    "patch_samples": 25,  # 250 Hz 下为 0.1 秒
    "width": 96,
    "attention_heads": 4,
    "domain_blocks": 2,
    "shared_blocks": 1,
    "ffn_ratio": 2,
    "dropout": 0.1,
}
