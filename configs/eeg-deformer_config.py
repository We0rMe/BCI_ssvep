# 本模型的训练超参数
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 100
OPTIMIZER = "adam"  # adam 或 sgd
SCHEDULER = "cosine"  # cosine 或 constant
WEIGHT_DECAY = 1e-5
SGD_MOMENTUM = 0.9
LEGACY_EEG_RUN_DIR = None
LEGACY_EEG_DATA_FILE = None


INPUT_TYPE = "time"  # time 为时域，complex_channels 沿通道拼接实虚部，complex 沿频率拼接
FFT_RESOLUTION = 0.2  # 频点间距（Hz）
SPECTRUM_HZ = [8.0, 64.0]  # 复频谱频段（Hz），包含两端

SETTINGS = {
    "depth": 3,  # HCT 块数
    "temporal_kernel": 25,  # 沿最后一轴的卷积核长度，单位为采样点或频点，须为奇数
    "num_filters": 64,
    "heads": 16,
    "head_dim": 16,
    "ff_dim": 16,
    "dropout": 0.5,
    "max_norm": 2.0,
}

# 历史时域配置只含模型参数，复频谱实验额外保存输入参数。
if INPUT_TYPE != "time":
    SETTINGS["input"] = {
        "input_type": INPUT_TYPE, 
        "fft_resolution": FFT_RESOLUTION, 
        "spectrum_hz": SPECTRUM_HZ,
        "include_upper_bin": True, 
        "fft_scale": 1.0,
    }
