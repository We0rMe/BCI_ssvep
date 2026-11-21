# 本模型的训练超参数
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 100
OPTIMIZER = "sgd"  # adam 或 sgd
SCHEDULER = "cosine"  # cosine 或 constant
WEIGHT_DECAY = 1e-5
SGD_MOMENTUM = 0.9


SETTINGS = {
    "input_type": "complex",  # complex、complex_channels、amplitude 或 time
    "fft_resolution": 0.2,
    "spectrum_hz": (8.0, 64.0),
    "include_upper_bin": True,
    "fft_scale": 1.0,  # FFT × 1 / 窗内采样点数
    "init_std": 0.01,
    "spatial_multiplier": 2,
    "encoder_blocks": 2,
    "kernel_size": 31,
    "head_multiplier": 6,
    "dropout": 0.5,
}
