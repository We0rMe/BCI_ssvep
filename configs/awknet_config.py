"""AWKNet 的跨被试 SSVEP 配置。"""

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 100
OPTIMIZER = "adam"
SCHEDULER = "cosine"
WEIGHT_DECAY = 1e-5
SGD_MOMENTUM = 0.9  # 共用训练接口读取此项，Adam 不使用

SETTINGS = {
    "wavelet_lr": 1e-4,
    "wavelet_penalty": 1.0,
    "use_d2": True,
    "kernel_sizes": (3, 5, 7),
    "filters_per_kernel": 8,
    "pool_size": 2,
    "grid_size": 3,
    "spline_order": 3,
    "grid_range": (-3.0, 3.0),  # 读出前对每个试次做 LayerNorm
    "init_std": 0.01,
}
