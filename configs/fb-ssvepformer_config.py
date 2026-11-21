from .ssvepformer_config import SETTINGS as SUBNETWORK_SETTINGS


# 本模型的训练超参数
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
# 各阶段轮数见下方 pretrain_epochs 和 joint_epochs
OPTIMIZER = "adam"  # adam 或 sgd
SCHEDULER = "cosine"  # cosine 或 constant
WEIGHT_DECAY = 1e-5
SGD_MOMENTUM = 0.9


SETTINGS = {
    **SUBNETWORK_SETTINGS,
    "subbands_hz": ((9.0, 80.0), (18.0, 80.0), (27.0, 80.0)),
    "lower_transition_hz": 2.0,
    "upper_stop_hz": 90.0,
    "passband_ripple_db": 0.5,
    "stopband_attenuation_db": 40.0,
    "pretrain_epochs": 100,  # 每个子网单独训练的轮数
    "joint_epochs": 20,  # 全网联合训练的轮数
}
