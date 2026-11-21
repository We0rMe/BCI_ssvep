"""同一次 Benchmark 实验中各模型共用的运行设置。"""

from data_preprocess.config import NUM_SUBJECTS


# 按此顺序训练；单次运行可通过 --models 覆盖。
# 可选模型：mtsnet、ssvepformer、fb-ssvepformer、ssvepformer_eKAN、
# spiketfnet、eeg-deformer、ssvepgpt。例如 MODELS = ("mtsnet", "eeg-deformer")
MODELS = ("eeg-deformer", "mtsnet", "ssvepformer_eKAN", "ssvepformer", "ssvepgpt")
RESULT_NAME = "benchmark_loso34"  # 本次实验的结果目录名，训练和评估均使用 results/<RESULT_NAME>/
DEVICE = "cuda"  # "cpu" 或 "cuda"
CUDA_DEVICE_IDS = (0,1)  # 双卡示例 (0, 1)，按 MODELS 顺序轮流分配，每个模型仅用一张卡

SEED = 42
LOSO_FOLDS = NUM_SUBJECTS  # 前 N 折，每折用全部非测试被试训练
DETERMINISTIC = True
ALLOW_TF32 = False
ITR_GAZE_SECONDS = 0.5  # ITR 选择时间等于窗长加此时长

GPU_WARMUP_BATCHES = 3
GPU_PROBE_BATCHES = 5
