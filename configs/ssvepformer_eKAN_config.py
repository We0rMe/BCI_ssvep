from .ssvepformer_config import SUBNETWORK_SETTINGS


# 本模型的训练超参数
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 100
OPTIMIZER = "sgd"  # adam 或 sgd
SCHEDULER = "constant"  # cosine 或 constant
WEIGHT_DECAY = 1e-3
SGD_MOMENTUM = 0.9


SETTINGS = {
    **SUBNETWORK_SETTINGS,
    "grid_size": 3,
    "spline_order": 3,
    "grid_range": (-1.0, 4.0),  # 覆盖 GELU 后的负值和 Dropout 放大的正值
}
