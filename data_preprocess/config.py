from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]

# 常用设置
BENCHMARK_DIR = PROJECT_DIR / "data" / "benchmark"  # 原始 MAT 文件目录
OUTPUT_FILE = PROJECT_DIR / "data_preprocess" / "benchmark.npz"  # 共用预处理文件
WINDOW_SECONDS = (0.5, 1.0)  # 同时准备的分类窗长，单位为秒

# Benchmark 原始数据属性
SAMPLING_RATE_HZ = 250
NUM_SUBJECTS = 35
NUM_CHANNELS = 64
TRIAL_SAMPLES = 1500
NUM_CLASSES = 40
NUM_BLOCKS = 6
CUE_SECONDS = 0.50
STIMULUS_SECONDS = 5.0

# 统一的分类窗与带通规则
CHANNELS = ("O1", "Oz", "O2", "PO3", "POz", "PO4", "Pz", "PO5", "PO6")
VISUAL_LATENCY_SECONDS = 0.14
BANDPASS_HZ = (8.0, 64.0)
BUTTERWORTH_ORDER = 4  # 巴特沃斯低通原型阶数
