# 独立的线性基线配方，与 RMS 版 SsvepGPT 使用相同的优化设置。
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 100
OPTIMIZER = "adam"
SCHEDULER = "cosine"
WEIGHT_DECAY = 1e-5  # Adam 的耦合 L2，作用于权重和偏置
SGD_MOMENTUM = 0.9


SETTINGS = {
    "rms_floor": 1e-12,
    "input_type": "complex",  # 每个电极先实部、后虚部
    "fft_resolution": 0.2,
    "spectrum_hz": (8.0, 64.0),
    "include_upper_bin": True,
    "fft_scale": 1.0,
}
