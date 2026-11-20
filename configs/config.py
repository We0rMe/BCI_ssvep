from data_preprocess.config import NUM_SUBJECTS, PROJECT_DIR


# 实验范围与保存位置
RESULT_NAME = "benchmark_ssvepformer_eKAN"  # 结果根目录名称，位于 results 下
MODELS = ("ssvepformer_eKAN", "ssvepformer",)  # 本次训练的模型
LOSO_FOLDS = NUM_SUBJECTS  # 依次留出 S1 至 S{LOSO_FOLDS}
SEED = 42  # 单次对照使用一个种子
VALIDATION_FRACTION = 0.20  # 每折从非测试被试中划出的验证比例

# 共用训练配方
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 200  # 单阶段模型的训练轮数
OPTIMIZER = "adam"  # adam 或 sgd，均使用余弦退火
WEIGHT_DECAY = 1e-5
SGD_MOMENTUM = 0.9  # 仅 SGD 使用

# 设备与精度
DEVICE = "cuda"  # cpu 或 cuda
CUDA_DEVICE_IDS = (0,1)  # (0,) 单卡；双卡且有多个模型时，每卡独立训练一个模型
DETERMINISTIC = True
ALLOW_TF32 = False

# 测试及兼容旧 EEG-Deformer 结果
ITR_GAZE_SECONDS = 0.5  # ITR 的选择时间 T = 窗长 + 0.5 s
LEGACY_EEG_RUN_DIR = None  # 需要兼容旧权重时填写旧实验根目录的路径
LEGACY_EEG_DATA_FILE = None  # 可指定旧 NPZ；默认从旧项目的 data_processed 目录查找

# GPU 冒烟测试
GPU_WARMUP_BATCHES = 3
GPU_PROBE_BATCHES = 5

RESULTS_DIR = PROJECT_DIR / "results"
