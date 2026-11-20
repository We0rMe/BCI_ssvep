from .ssvepformer_config import SETTINGS as SUBNETWORK_SETTINGS


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
