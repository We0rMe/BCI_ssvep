from data_preprocess.config import NUM_SUBJECTS, OUTPUT_FILES

# 可选模型：mtsnet、ssvepformer、fb-ssvepformer、ssvepformer_eKAN、
MODELS = ("ssvepformer","ssvepformer_eKAN","fb-ssvepformer")
PREPROCESS_IDS = (1,)  # (1,) 先截窗；(2,) 先滤波；(1, 2) 依次独立运行
RESULT_NAME = "ssvepformer_series_spectralnorm"  # 本次实验的结果目录名，训练和评估均使用 results/<RESULT_NAME>/
DEVICE = "cuda"  # "cpu" 或 "cuda"
CUDA_DEVICE_IDS = (0,1)  # 双卡示例 (0, 1)，按 MODELS 顺序轮流分配，每个模型仅用一张卡

SEED = 42
LOSO_FOLDS = NUM_SUBJECTS  # 前 N 折，每折用全部非测试被试训练
DETERMINISTIC = True
ALLOW_TF32 = False
ITR_GAZE_SECONDS = 0.5  # ITR 选择时间等于窗长加此时长

GPU_WARMUP_BATCHES = 3
GPU_PROBE_BATCHES = 5


def experiment_datasets(base_dir):
    ids = (PREPROCESS_IDS,) if isinstance(PREPROCESS_IDS, int) else tuple(PREPROCESS_IDS)
    if (not ids or any(type(i) is not int or i not in OUTPUT_FILES for i in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("PREPROCESS_IDS 须选 1、2 或两者，且不能重复")
    for method in ids:
        directory = base_dir if method == 1 else base_dir / "preprocess_2"
        yield method, OUTPUT_FILES[method], directory
