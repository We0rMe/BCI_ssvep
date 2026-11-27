import argparse
import csv
import hashlib
import io
import json
import math
import multiprocessing
import os
from importlib import import_module
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from filelock import FileLock
from matplotlib.figure import Figure
from torch import nn
from torch.nn.functional import cross_entropy
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader, TensorDataset

from configs import config
from data_preprocess import config as data_config
from data_preprocess.preprocess import start_sample, window_samples


EPOCH_COLUMNS = (
    "epoch", "stage", "stage_epoch", "lr", "train_loss", "train_acc",
    "train_seconds",
)


def training_settings(model_cfg):
    settings = {
        "SEED": config.SEED, "LOSO_FOLDS": config.LOSO_FOLDS,
        "BATCH_SIZE": model_cfg.BATCH_SIZE, "OPTIMIZER": model_cfg.OPTIMIZER,
        "LEARNING_RATE": model_cfg.LEARNING_RATE, "WEIGHT_DECAY": model_cfg.WEIGHT_DECAY,
        "SGD_MOMENTUM": model_cfg.SGD_MOMENTUM, "ITR_GAZE_SECONDS": config.ITR_GAZE_SECONDS,
        "SCHEDULER": model_cfg.SCHEDULER,
    }
    if hasattr(model_cfg, "EPOCHS"):
        settings["EPOCHS"] = model_cfg.EPOCHS
    if model_cfg.OPTIMIZER == "adamw":
        settings.update(ADAM_BETAS=model_cfg.ADAM_BETAS, ADAM_EPS=model_cfg.ADAM_EPS)
    if model_cfg.SCHEDULER == "warmup_cosine":
        settings["WARMUP_EPOCHS"] = model_cfg.WARMUP_EPOCHS
    return settings


def selected_models(models=None):
    if models is None:
        models = config.MODELS
    names = (models,) if isinstance(models, str) else tuple(models)
    if not names or len(set(names)) != len(names):
        raise ValueError("--models 至少选一个模型，且不能重复")
    return names


def experiment_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", help="临时覆盖 configs/config.py 的 MODELS，按训练顺序列出")
    parser.add_argument("--run-dir", type=Path, help="覆盖默认实验目录 results/<RESULT_NAME>")
    args = parser.parse_args()
    names = selected_models(args.models)
    directory = args.run_dir or data_config.PROJECT_DIR / "results" / config.RESULT_NAME
    return names, directory


def model_config(name):
    return import_module(f"configs.{name}_config")


def model_module(name):
    if not name.replace("-", "_").isidentifier():
        raise ValueError(f"模型名 {name} 须对应 models 下的一个脚本")
    return import_module(f"models.{name}")


def archive_method(archive, expected=None):
    # 早期 benchmark.npz 只支持先截窗，保留其文件格式和 SHA-256。
    method = np.asarray(archive.get("preprocess_id", 1)).item()
    if method not in data_config.OUTPUT_FILES or (expected is not None and expected != method):
        raise ValueError(f"预处理文件编号 {method} 与配置不符（期望 {expected}）")
    return int(method)


def check_archive(archive, counts):
    expected = {
        "subject_ids": list(range(1, data_config.NUM_SUBJECTS + 1)),
        "channels": list(data_config.CHANNELS),
        "sampling_rate_hz": data_config.SAMPLING_RATE_HZ,
        "window_start_sample": start_sample(),
        "bandpass_hz": list(data_config.BANDPASS_HZ),
        "butterworth_order": data_config.BUTTERWORTH_ORDER,
    }
    mismatched = [key for key, value in expected.items() if archive[key].tolist() != value]
    missing = {f"{kind}_{count}" for kind in ("eeg", "raw") for count in counts}.difference(archive.files)
    if archive_method(archive) == 2:
        missing.update({f"raw_trials_s{s}" for s in archive["subject_ids"]}.difference(archive.files))
    if mismatched or missing:
        fields = ", ".join(mismatched + sorted(missing))
        raise ValueError(f"预处理文件与当前配置不符（{fields}），请重新执行预处理")


def read_archive(source):
    digest = hashlib.file_digest(source, "sha256").hexdigest()
    source.seek(0)
    return np.load(source, allow_pickle=False), digest


def subject_loader(inputs, labels, subject_ids, selected_ids, batch_size, device, generator=None):
    indices = {int(subject_id): index for index, subject_id in enumerate(subject_ids)}
    selection = [indices[subject_id] for subject_id in selected_ids]
    values = [torch.from_numpy(array[selection].reshape(-1, *array.shape[2:])) for array in inputs]
    targets = torch.from_numpy(np.tile(labels, len(selection)))
    return DataLoader(
        TensorDataset(*values, targets), batch_size=batch_size, shuffle=generator is not None,
        pin_memory=device.type == "cuda", generator=generator,
    )


def input_shapes(inputs):
    return tuple(array.shape[2:] for array in inputs)


def prepare_inputs(module, archive, count, options):
    inputs = module.prepare_inputs(archive, count, options)
    expected = len(archive["subject_ids"]), len(archive["labels"])
    if any(array.shape[:2] != expected for array in inputs):
        raise ValueError(f"{module.__name__} 的输入须与共用数据中的被试和试次对应")
    return inputs


def set_seed(seed, cuda_ids):
    torch.random.default_generator.manual_seed(seed)
    for index in cuda_ids:
        with torch.cuda.device(index):
            torch.cuda.manual_seed(seed)


def parallel_model(module, model, cuda_ids):
    if len(cuda_ids) == 2:
        wrapper = getattr(module, "PARALLEL_WRAPPER", nn.DataParallel)
        return wrapper(model, device_ids=cuda_ids)
    return model


def make_optimizer(model, settings):
    options = {"lr": settings["LEARNING_RATE"], "weight_decay": settings["WEIGHT_DECAY"]}
    parameters = model.optimizer_groups() if hasattr(model, "optimizer_groups") else model.parameters()
    if settings["OPTIMIZER"] == "sgd":
        return SGD(parameters, momentum=settings["SGD_MOMENTUM"], **options)
    if settings["OPTIMIZER"] == "adamw":
        return AdamW(parameters, betas=settings["ADAM_BETAS"], eps=settings["ADAM_EPS"], **options)
    return Adam(parameters, **options)


def make_scheduler(optimizer, settings, epochs):
    if settings["SCHEDULER"] == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs)
    if settings["SCHEDULER"] == "warmup_cosine":
        warmup = settings["WARMUP_EPOCHS"]
        if not 1 <= warmup < epochs:
            raise ValueError("预热轮数须至少为 1 且小于总轮数")

        def lr_factor(index):
            if index < warmup:
                return (index + 1) / warmup
            progress = (index + 1 - warmup) / (epochs + 1 - warmup)
            return (1 + math.cos(math.pi * progress)) / 2

        return LambdaLR(optimizer, lr_factor)
    return None


def training_plan(module, options, epochs):
    # 每项包含阶段名、训练轮数和子带编号。None 表示使用完整输入训练整个模型。
    if hasattr(module, "training_plan"):
        return module.training_plan(options)
    return [("main", epochs, None)]


def prediction_logits(output):
    return output["logits"] if isinstance(output, dict) else output


def batch_step(network, batch, device, optimizer=None, auxiliary_loss=None, band=None):
    inputs, targets = batch[:-1], batch[-1]
    if band is not None:
        inputs = (inputs[0][:, band],)
    inputs = tuple(value.to(device, non_blocking=device.type == "cuda") for value in inputs)
    targets = targets.to(device, non_blocking=device.type == "cuda")
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    output = network(*inputs)
    logits = prediction_logits(output)
    loss = cross_entropy(logits, targets) if auxiliary_loss is None else auxiliary_loss(output, targets)
    if optimizer is not None:
        loss.backward()
        optimizer.step()
    correct = (logits.detach().argmax(dim=1) == targets).sum()
    return loss.detach(), correct, targets.numel()


def run_epoch(network, loader, device, optimizer, auxiliary_loss, band):
    training = optimizer is not None
    network.train(training)
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    examples = 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            loss, matches, size = batch_step(network, batch, device, optimizer, auxiliary_loss, band)
            loss_sum += loss * size
            correct += matches
            examples += size
    return loss_sum.item() / examples, correct.item() / examples


def write_epoch(fold_dir, history, state):
    artifacts = fold_dir / "artifacts"
    # 先写日志。若 checkpoint 尚未提交，续跑时按 checkpoint 的 epoch 截回日志。
    save_history(artifacts / "training.log", history)
    save_checkpoint(artifacts / "checkpoint.pt", state)
    row = history[-1]
    print(f"[{fold_dir.parent.name}/{fold_dir.name}] {row['stage']} {row['stage_epoch']:03d} | "
          f"train loss {row['train_loss']:.4f} ACC {row['train_acc']:.2%} | "
          f"LR {row['lr']:.3g} | 训练 {row['train_seconds']:.2f}s", flush=True)
    save_curve(fold_dir / "train.png", history)


def fold_summary(history):
    return history[-1]["train_acc"], sum(row["train_seconds"] for row in history)


def train_fold(module, options, plan, settings, inputs, labels, subject_ids, split, fold_dir, device, cuda_ids):
    artifacts = fold_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    checkpoint_file = artifacts / "checkpoint.pt"
    log_file = artifacts / "training.log"
    state = torch.load(checkpoint_file, map_location="cpu", weights_only=True) if checkpoint_file.exists() else None
    history = []
    if state is not None:
        history = load_history(log_file, state["epoch"])
        save_history(log_file, history)
        save_curve(fold_dir / "train.png", history)
        if state["complete"]:
            return fold_summary(history)

    set_seed(settings["SEED"], cuda_ids)
    shuffle = torch.Generator().manual_seed(settings["SEED"])
    model = module.build_model(input_shapes(inputs), data_config.NUM_CLASSES, options).to(device)
    train_loader = subject_loader(inputs, labels, subject_ids, split["train"], settings["BATCH_SIZE"], device, shuffle)
    auxiliary_loss = None
    if hasattr(module, "joint_loss"):
        auxiliary_loss = lambda output, targets: module.joint_loss(output, targets, options)

    first_stage = 0
    if state is not None:
        model.load_state_dict(state["model"])
        shuffle.set_state(state["shuffle_rng"])
        torch.set_rng_state(state["torch_rng"])
        for index, rng in zip(cuda_ids, state["cuda_rng"], strict=True):
            torch.cuda.set_rng_state(rng, device=index)
        first_stage = state["stage_index"]
        if state["stage_epoch"] == plan[first_stage][1]:
            first_stage += 1
        print(f"[{fold_dir.parent.name}/{fold_dir.name}] 已恢复 {state['epoch']} 个 epoch", flush=True)

    for stage_index in range(first_stage, len(plan)):
        stage, epochs, band = plan[stage_index]
        active = model if band is None else model.subnetworks[band]
        network = parallel_model(module, active, cuda_ids)
        optimizer = make_optimizer(active, settings)
        scheduler = make_scheduler(optimizer, settings, epochs)
        done = 0
        if state is not None and stage_index == state["stage_index"]:
            optimizer.load_state_dict(state["optimizer"])
            if scheduler is not None:
                scheduler.load_state_dict(state["scheduler"])
            done = state["stage_epoch"]

        for stage_epoch in range(done + 1, epochs + 1):
            lr = optimizer.param_groups[0]["lr"]
            started = perf_counter()
            train_loss, train_acc = run_epoch(network, train_loader, device, optimizer, auxiliary_loss, band)
            train_seconds = perf_counter() - started
            if not math.isfinite(train_loss):
                raise FloatingPointError(f"{stage} 第 {stage_epoch} 轮出现非有限损失，已保留上一断点")
            if scheduler is not None:
                scheduler.step()
            epoch = len(history) + 1
            history.append({
                "epoch": epoch, "stage": stage, "stage_epoch": stage_epoch, "lr": lr,
                "train_loss": train_loss, "train_acc": train_acc,
                "train_seconds": train_seconds,
            })
            write_epoch(fold_dir, history, {
                "epoch": epoch, "stage_index": stage_index, "stage_epoch": stage_epoch,
                "complete": stage_index == len(plan) - 1 and stage_epoch == epochs,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "shuffle_rng": shuffle.get_state(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": [torch.cuda.get_rng_state(index) for index in cuda_ids],
            })

    return fold_summary(history)


def fold_directory(model_dir, samples, subject):
    return model_dir / f"{samples}pt_s{subject:02d}"


def write_training_report(model_dir, name, fs, results):
    lines = [f"# {name} 训练汇总", "", "| 窗长 | 完成折数 | 末轮训练 ACC | 累计训练时间（s） |",
             "| --- | ---: | ---: | ---: |"]
    for count, scores, seconds in results:
        acc = f"{100 * np.mean(scores):.2f}% ± {100 * np.std(scores):.2f} pp"
        lines.append(f"| {count / fs:g} s | {len(scores)} | {acc} | {seconds:.2f} |")
    lines.extend(("", "末轮训练 ACC 使用折间总体标准差。跨模型的测试指标见 [实验报告](../report.md)。"))
    save_text(model_dir / "report.md", "\n".join(lines) + "\n")


def model_devices(names, cuda_ids):
    if not cuda_ids:
        return {name: [] for name in names}
    return {name: [cuda_ids[index % len(cuda_ids)]] for index, name in enumerate(names)}


def train_model(name, manifest, archive, run_dir, device):
    module = model_module(name)
    options = manifest["models"][name]
    plan = manifest["training_plans"][name]
    settings = manifest["model_runs"][name]["settings"]
    cuda_ids = manifest["model_devices"][name]
    subject_ids, labels = archive["subject_ids"], archive["labels"]
    model_dir = run_dir / name
    results = []
    print(f"{name} 使用 {device}", flush=True)
    for count in manifest["window_samples"]:
        inputs = prepare_inputs(module, archive, count, options)
        scores, seconds = [], 0.0
        for split in manifest["model_runs"][name]["splits"]:
            folder = fold_directory(model_dir, count, split["test"])
            print(f"\n{name} / {folder.name}", flush=True)
            acc, elapsed = train_fold(module, options, plan, settings, inputs, labels, subject_ids,
                                      split, folder, device, cuda_ids)
            scores.append(acc)
            seconds += elapsed
        results.append((count, scores, seconds))
        del inputs
    write_training_report(model_dir, name, manifest["sampling_rate_hz"], results)


def train_on_gpu(gpu_id, names, data_file, run_dir, manifest):
    with data_file.open("rb") as source:
        archive, digest = read_archive(source)
        with archive:
            if digest != manifest["data_sha256"]:
                raise ValueError("训练期间预处理文件发生变化")
            for name in names:
                device, _ = configure_runtime((gpu_id,))
                train_model(name, manifest, archive, run_dir, device)


def train_on_two_gpus(data_file, run_dir, manifest, cuda_ids):
    context = multiprocessing.get_context("spawn")
    workers = []
    for gpu_id in cuda_ids:
        names = [name for name, devices in manifest["model_devices"].items() if devices == [gpu_id]]
        workers.append(context.Process(target=train_on_gpu, name=f"cuda-{gpu_id}",
                                       args=(gpu_id, names, data_file, run_dir, manifest)))
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join()
    failed = [worker.name for worker in workers if worker.exitcode != 0]
    if failed:
        raise RuntimeError(f"模型训练进程退出异常（{', '.join(failed)}），可从已写入的断点恢复")


def train_experiment(data_file, run_dir, names=None, preprocess_id=None):
    names = selected_models(names)
    modules = {name: model_module(name) for name in names}
    configs = {name: model_config(name) for name in names}
    model_options = {name: cfg.SETTINGS for name, cfg in configs.items()}
    recipes = {name: training_settings(cfg) for name, cfg in configs.items()}
    plans = {name: training_plan(modules[name], model_options[name], recipes[name].get("EPOCHS")) for name in names}
    for name, recipe in recipes.items():
        if recipe["OPTIMIZER"] not in ("adam", "adamw", "sgd") or recipe["SCHEDULER"] not in ("cosine", "constant", "warmup_cosine"):
            raise ValueError(f"{name} 的优化器或学习率调度器无效")
        if not plans[name] or any(not isinstance(n, int) or n < 1 for _, n, _ in plans[name]):
            raise ValueError(f"{name} 训练阶段的轮数须为正整数")

    _, cuda_ids = configure_runtime()
    devices = model_devices(names, cuda_ids)
    counts = window_samples(data_config.WINDOW_SECONDS)
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    with FileLock(artifacts / "run.lock", timeout=0), data_file.open("rb") as source:
        archive, digest = read_archive(source)
        with archive:
            method = archive_method(archive, preprocess_id)
            check_archive(archive, counts)
            subject_ids = archive["subject_ids"]
            fs = archive["sampling_rate_hz"].item()
            model_runs = {}
            for name in names:
                device, ids = configure_runtime(devices[name])
                model_runs[name] = {
                    "settings": recipes[name],
                    "splits": loso_splits(subject_ids, recipes[name]["LOSO_FOLDS"]),
                    "runtime": runtime_details(device, ids),
                }
            manifest = {
                "protocol": "loso_final_epoch", "model_runs": model_runs,
                "data_sha256": digest, "sampling_rate_hz": fs, "num_classes": data_config.NUM_CLASSES,
                "window_samples": counts, "model_order": list(names), "models": model_options,
                "training_plans": plans,
                "model_devices": devices,
            }
            if method == 2:
                manifest["preprocess_id"] = method
            save_manifest(artifacts / "run.json", manifest)
            report = run_dir / "report.md"
            if not report.exists():
                links = [f"- [{name} 的训练汇总]({name}/report.md)" for name in names]
                save_text(report, "# 实验汇总\n\n训练完成后运行 `evaluate.py` 汇总留出被试的 ACC 和 ITR。\n\n"
                          + "\n".join(links) + "\n")

            if len(cuda_ids) == 2 and len(names) > 1:
                train_on_two_gpus(data_file, run_dir, manifest, cuda_ids)
            else:
                for name in names:
                    device, _ = configure_runtime(devices[name])
                    train_model(name, manifest, archive, run_dir, device)
    print(f"训练汇总已保存到 {run_dir / 'report.md'}", flush=True)


def loso_splits(subject_ids, num_folds):
    subjects = sorted(map(int, subject_ids))
    if len(subjects) < 2 or len(set(subjects)) != len(subjects):
        raise ValueError("LOSO 至少需要两个不同的被试，且编号不能重复")
    if not 1 <= num_folds <= len(subjects):
        raise ValueError("LOSO 折数须在 1 至被试总数之间")
    return [{"test": test, "train": [subject for subject in subjects if subject != test]}
            for test in subjects[:num_folds]]


def configure_runtime(device_ids=None):
    if config.DEVICE not in ("cpu", "cuda"):
        raise ValueError('DEVICE 须为 "cpu" 或 "cuda"')
    device = torch.device(config.DEVICE)
    ids = config.CUDA_DEVICE_IDS if device_ids is None else device_ids
    cuda_ids = list(ids) if config.DEVICE == "cuda" else []
    if device.type == "cuda":
        if config.DETERMINISTIC:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        count = torch.cuda.device_count()
        if (len(cuda_ids) not in (1, 2) or len(set(cuda_ids)) != len(cuda_ids)
                or any(not 0 <= index < count for index in cuda_ids)):
            raise ValueError(f"当前有 {count} 张可见 CUDA 卡，CUDA_DEVICE_IDS 须指定一张或两张可用卡")
        device = torch.device("cuda", cuda_ids[0])
        torch.cuda.set_device(device)
        precision = "tf32" if config.ALLOW_TF32 else "ieee"
        torch.backends.cuda.matmul.fp32_precision = precision
        torch.backends.cudnn.conv.fp32_precision = precision

    torch.use_deterministic_algorithms(config.DETERMINISTIC)
    torch.backends.cudnn.deterministic = config.DETERMINISTIC
    torch.backends.cudnn.benchmark = bool(cuda_ids) and not config.DETERMINISTIC
    return device, cuda_ids


def runtime_details(device, cuda_ids):
    return {
        "torch_version": str(torch.__version__),
        "device": str(device),
        "cuda_ids": cuda_ids,
        "gpu_names": [torch.cuda.get_device_name(index) for index in cuda_ids],
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if cuda_ids else None,
        "cudnn_enabled": torch.backends.cudnn.enabled if cuda_ids else None,
        "precision": "float32",
        "cuda_matmul_precision": torch.backends.cuda.matmul.fp32_precision if cuda_ids else None,
        "cudnn_conv_precision": torch.backends.cudnn.conv.fp32_precision if cuda_ids else None,
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace": os.getenv("CUBLAS_WORKSPACE_CONFIG") if cuda_ids else None,
    }


def atomic_write(path, write):
    # 同一实验目录由文件锁保护，临时文件和目标文件位于同一个目录。
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("wb") as file:
            write(file)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def save_text(path, text):
    atomic_write(path, lambda file: file.write(text.encode("utf-8")))


def save_manifest(path, settings):
    text = json.dumps(settings, indent=2, ensure_ascii=False) + "\n"
    if not path.exists():
        save_text(path, text)
    elif path.read_text(encoding="utf-8") != text:
        raise ValueError("实验目录已有不同的数据、环境或配置，请恢复原设置或更换 RESULT_NAME 或 --run-dir")


def save_history(path, history):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=EPOCH_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(history)
    save_text(path, buffer.getvalue())


def load_history(path, committed_epochs):
    with path.open(encoding="utf-8", newline="") as file:
        records = list(csv.DictReader(file))[:committed_epochs]
    if [int(row["epoch"]) for row in records] != list(range(1, committed_epochs + 1)):
        raise ValueError(f"{path} 缺少 checkpoint 对应的 epoch 记录")
    return [
        {key: int(value) if key in ("epoch", "stage_epoch") else value if key == "stage" else float(value) for key, value in row.items()}
        for row in records
    ]


def save_curve(path, history):
    figure = Figure(figsize=(9, 3.4), dpi=125, layout="constrained")
    loss_axis, acc_axis = figure.subplots(1, 2)
    epochs = [record["epoch"] for record in history]
    marker = "." if len(epochs) == 1 else None
    loss_axis.plot(epochs, [row["train_loss"] for row in history], label="Train", marker=marker)
    acc_axis.plot(epochs, [100 * row["train_acc"] for row in history], label="Train", marker=marker)
    loss_axis.set(xlabel="Epoch", ylabel="Loss")
    acc_axis.set(xlabel="Epoch", ylabel="Accuracy (%)", ylim=(0, 100))
    for axis in (loss_axis, acc_axis):
        axis.grid(alpha=0.2)
        axis.legend()
    atomic_write(path, lambda file: figure.savefig(file, format="png"))
    figure.clear()


def save_checkpoint(path, state):
    atomic_write(path, lambda file: torch.save(state, file))


if __name__ == "__main__":
    names, directory = experiment_args()
    for method, data_file, run_dir in config.experiment_datasets(directory):
        train_experiment(data_file, run_dir, names, method)
