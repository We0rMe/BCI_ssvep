# 本模型的训练超参数
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 100
OPTIMIZER = "adam"  # adam 或 sgd
SCHEDULER = "cosine"  # cosine 或 constant
WEIGHT_DECAY = 1e-5
SGD_MOMENTUM = 0.9


SETTINGS = {
    "normalize_trial_rms": True,  # 先逐试次统一幅值，再从同一波形计算复频谱
    "input_type": "complex",  # 每个电极的整窗频谱，先实部、后虚部
    "fft_resolution": 0.2,
    "spectrum_hz": (8.0, 64.0),
    "include_upper_bin": True,
    "fft_scale": 1.0,
    "spectrum_bands": 4,  # 每个电极的频域 token 数；1 对应原来的整谱投影
    "patch_samples": 25,  # 250 Hz 下为 0.1 秒
    "width": 48,
    "attention_heads": 4,
    "domain_blocks": 2,
    "shared_blocks": 1,
    "ffn_ratio": 2,
    "dropout": 0.1,
}
