"""Benchmark 固定窗 iFuzzyTL。学习率是单模型、单卡的峰值。"""

BATCH_SIZE = 64
LEARNING_RATE = 5e-4
EPOCHS = 800
OPTIMIZER = "adamw"
ADAM_BETAS = (0.9, 0.95)
ADAM_EPS = 1e-8
WEIGHT_DECAY = 0.05
SGD_MOMENTUM = 0.9  # 共用配置接口读取，AdamW 不使用
SCHEDULER = "warmup_cosine"
WARMUP_EPOCHS = 20

SETTINGS = {
    "spatial_rules": 10,
    "temporal_rules": 10,
    "dropout": 0.3,
    # 初始宽度按特征数的四次方根放大，缓解长窗初期规则权重过于集中的问题。
    "spatial_width_scale": 2.0,
    "temporal_width_scale": 1.0,
}
