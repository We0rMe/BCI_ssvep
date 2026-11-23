# 本模型的训练超参数
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 100
OPTIMIZER = "sgd"  # adam 或 sgd
SCHEDULER = "constant"  # 补充实验：SGD 全程使用固定学习率
WEIGHT_DECAY = 1e-4
SGD_MOMENTUM = 0.9


SETTINGS = {
    "standardize_trials": True,  # 先对每个滤波后的 EEG 窗做 z-score，再计算 FFT
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
