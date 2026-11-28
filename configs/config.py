from data_preprocess.config import NUM_SUBJECTS, OUTPUT_FILES

MODELS = ("ifuzzytl",)
WINDOW_MODE = "random"  # fixed 固定前置窗；random 每轮重新取窗
RANDOM_START_SECONDS = 0.64  # 相对整条 trial；设为 0.64 可排除刺激开始后的 140 ms
RANDOM_WINDOWS_PER_EPOCH = 12000
PREPROCESS_IDS = (1,)  # (1,) 先截窗；(2,) 先滤波；(1, 2) 依次独立运行
RESULT_NAME = "ifuzzytl_random_window"  # 训练和评估均使用 results/<RESULT_NAME>/
DEVICE = "cuda"  # "cpu" 或 "cuda"
CUDA_DEVICE_IDS = (0,)  # 双卡示例 (0, 1)，按 MODELS 顺序轮流分配，每个模型仅用一张卡

SEED = 42
LOSO_FOLDS = NUM_SUBJECTS  # 前 N 折，每折用全部非测试被试训练
DETERMINISTIC = True
ALLOW_TF32 = False
ITR_GAZE_SECONDS = 0.5  # ITR 选择时间等于窗长加此时长

GPU_WARMUP_BATCHES = 3
GPU_PROBE_BATCHES = 5


def experiment_datasets(base_dir):
    if WINDOW_MODE not in ("fixed", "random"):
        raise ValueError("WINDOW_MODE 须为 fixed 或 random")
    ids = (PREPROCESS_IDS,) if isinstance(PREPROCESS_IDS, int) else tuple(PREPROCESS_IDS)
    if (not ids or any(type(i) is not int or i not in OUTPUT_FILES for i in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("PREPROCESS_IDS 须选 1、2 或两者，且不能重复")
    root = base_dir / "random_windows" if WINDOW_MODE == "random" else base_dir
    for method in ids:
        directory = root if method == 1 else root / "preprocess_2"
        yield method, OUTPUT_FILES[2 if WINDOW_MODE == "random" else method], directory
