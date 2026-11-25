import csv
import io
import json
import math
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock

from configs import config
from data_preprocess import config as data_config
from data_preprocess.preprocess import window_samples
from train import (
    archive_method, atomic_write, check_archive, configure_runtime, fold_directory, input_shapes, loso_splits, model_module,
    experiment_args, model_config, model_devices, parallel_model, prediction_logits, prepare_inputs, read_archive, runtime_details, save_text,
    subject_loader, training_settings,
)


TEST_COLUMNS = ("samples", "subject_id", "seed", "correct", "trials", "acc_percent", "itr_bits_min")
LEGACY_MODEL_KEYS = {
    "depth": "HCT_DEPTH", "temporal_kernel": "TEMPORAL_KERNEL", "num_filters": "NUM_FILTERS",
    "heads": "ATTENTION_HEADS", "head_dim": "ATTENTION_HEAD_DIM", "ff_dim": "FEEDFORWARD_DIM",
    "dropout": "MODEL_DROPOUT", "max_norm": "ENCODER_MAX_NORM",
}


def itr(accuracy, num_classes, selection_seconds):
    if accuracy <= 1 / num_classes:
        return 0.0
    bits = math.log2(num_classes)
    if accuracy < 1:
        error = 1 - accuracy
        bits += accuracy * math.log2(accuracy) + error * math.log2(error / (num_classes - 1))
    return 60 * bits / selection_seconds


@torch.inference_mode()
def predict(network, loader, device):
    network.eval()
    outputs = []
    for batch in loader:
        inputs = [tensor.to(device, non_blocking=device.type == "cuda") for tensor in batch[:-1]]
        outputs.append(prediction_logits(network(*inputs)).cpu())
    logits = torch.cat(outputs).numpy()
    if not np.isfinite(logits).all():
        raise FloatingPointError("测试 logits 出现非有限值")
    return logits


def write_test_details(model_dir, name, rows, predictions):
    artifacts = model_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=TEST_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    save_text(artifacts / "test_by_subject.csv", csv_buffer.getvalue())
    atomic_write(artifacts / "predictions.npz", lambda file: np.savez_compressed(file, **predictions))

    report_path = model_dir / "report.md"
    report = report_path.read_text(encoding="utf-8") if report_path.exists() else f"# {name}\n"
    report = report.split("\n## 测试详情\n", 1)[0].rstrip()
    report += ("\n\n## 测试详情\n\n"
               "[逐被试 ACC 与 ITR](artifacts/test_by_subject.csv) · "
               "[预测 logits](artifacts/predictions.npz) · [跨模型测试汇总](../report.md)\n")
    save_text(report_path, report)


def evaluate_model(name, archive, plan, checkpoint_dir, output_dir, device, cuda_ids, legacy=False):
    module = model_module(name)
    options = plan["models"][name]
    settings = plan["settings"]
    labels, subject_ids = archive["labels"], archive["subject_ids"]
    fs, num_classes = plan["sampling_rate_hz"], plan["num_classes"]
    rows = []
    predictions = {"labels": labels, "block_ids": archive["block_ids"]}
    for count in plan["window_samples"]:
        network, inputs = None, None
        all_logits, tested = [], []
        for split in plan["splits"]:
            subject = split["test"]
            checkpoint = fold_directory(checkpoint_dir, count, subject) / "artifacts" / "checkpoint.pt"
            if not checkpoint.exists():
                continue
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            complete = state["epoch"] == settings["EPOCHS"] if legacy else state["complete"]
            if not complete:
                continue
            if network is None:
                inputs = prepare_inputs(module, archive, count, options)
                model = module.build_model(input_shapes(inputs), num_classes, options).to(device)
                network = parallel_model(module, model, cuda_ids)
            weight_key = "model" if plan.get("protocol") == "loso_final_epoch" else "best_model"
            model.load_state_dict(state[weight_key])
            loader = subject_loader(inputs, labels, subject_ids, [subject], settings["BATCH_SIZE"], device)
            logits = predict(network, loader, device)
            correct = int((logits.argmax(axis=1) == labels).sum())
            accuracy = correct / len(labels)
            rate = itr(accuracy, num_classes, count / fs + settings["ITR_GAZE_SECONDS"])
            rows.append({"samples": count, "subject_id": subject, "seed": settings["SEED"], "correct": correct,
                         "trials": len(labels), "acc_percent": 100 * accuracy, "itr_bits_min": rate})
            tested.append(subject)
            all_logits.append(logits)
            print(f"{output_dir.name} / {count}pt / S{subject:02d}  ACC {accuracy:.2%}  ITR {rate:.2f}", flush=True)
        if tested:
            predictions[f"subject_ids_{count}"] = np.asarray(tested)
            predictions[f"logits_{count}"] = np.stack(all_logits)

    write_test_details(output_dir, output_dir.name, rows, predictions)
    return rows


def load_legacy_plan(run_dir, settings):
    path = run_dir / "artifacts" / "run.json"
    original = json.loads(path.read_text(encoding="utf-8"))
    old_settings = original["settings"]
    recipe = {key: old_settings[key] for key in (
        "SEED", "LOSO_FOLDS", "VALIDATION_FRACTION", "EPOCHS", "BATCH_SIZE", "LEARNING_RATE", "WEIGHT_DECAY",
    )}
    recipe.update({"OPTIMIZER": "adam", "ITR_GAZE_SECONDS": settings["ITR_GAZE_SECONDS"]})
    options = {key: old_settings[value] for key, value in LEGACY_MODEL_KEYS.items()}
    if "input" in original:
        options["input"] = original["input"]
    return {**original, "settings": recipe, "num_classes": old_settings["NUM_CLASSES"],
            "models": {"eeg-deformer": options}}


def legacy_differences(old, new, old_data, shared_data):
    differences = []
    recipe_keys = ("SEED", "BATCH_SIZE", "LEARNING_RATE", "WEIGHT_DECAY", "EPOCHS", "OPTIMIZER")
    if any(old["settings"][key] != new["settings"][key] for key in recipe_keys):
        differences.append("训练配方")
    if new.get("protocol") == "loso_final_epoch":
        differences.append("权重选择（旧实验按验证准确率，新实验取末轮）")
    if old["splits"] != new["splits"]:
        differences.append("被试划分")
    old_options = old["models"]["eeg-deformer"]
    new_options = new["models"]["eeg-deformer"]
    if old_options.get("input") != new_options.get("input"):
        differences.append("EEG-Deformer 输入表示或频谱设置")
    if any(old_options[key] != new_options[key] for key in LEGACY_MODEL_KEYS):
        differences.append("EEG-Deformer 模型参数")
    fields = ("subject_ids", "labels", "channels", "block_ids", "sampling_rate_hz", "window_start_sample",
              "bandpass_hz", "butterworth_order", "frequencies_hz", "phases_rad")
    if (old["window_samples"] != new["window_samples"] or old["num_classes"] != new["num_classes"]
            or any(not np.array_equal(old_data[field], shared_data[field]) for field in fields)
            or any(not np.array_equal(old_data[f"eeg_{count}"], shared_data[f"eeg_{count}"])
                   for count in new["window_samples"])):
        differences.append("预处理数据或窗长")
    precision_keys = ("precision", "deterministic", "cuda_matmul_precision", "cudnn_conv_precision")
    assignments = new.get("model_devices", {})
    reference_ids = assignments.get("eeg-deformer", next(iter(assignments.values()), new["runtime"]["cuda_ids"]))
    if (any(old["runtime"][key] != new["runtime"][key] for key in precision_keys)
            or len(old["runtime"]["cuda_ids"]) != len(reference_ids)):
        differences.append("训练精度、确定性或并行设置")
    return differences


def evaluate_legacy(run_dir, legacy_run, legacy_data_file, shared, reference, device, cuda_ids):
    old = load_legacy_plan(legacy_run, reference["settings"])
    old_file = (Path(legacy_data_file) if legacy_data_file is not None
                else legacy_run.parent.parent / "data_processed" / "benchmark.npz")
    with old_file.open("rb") as source:
        archive, digest = read_archive(source)
        with archive:
            if digest != old["data_sha256"]:
                raise ValueError(f"旧预处理文件的指纹与旧实验记录不符（{old_file}）")
            differences = legacy_differences(old, reference, archive, shared)
            output_dir = run_dir / "eeg-deformer-legacy"
            rows = evaluate_model("eeg-deformer", archive, old, legacy_run, output_dir, device, cuda_ids, legacy=True)
            save_text(output_dir / "artifacts" / "source.json", json.dumps({
                "run_dir": str(legacy_run), "data_file": str(old_file), "plan": old, "protocol_differences": differences,
            }, indent=2, ensure_ascii=False) + "\n")
    return "eeg-deformer-legacy", old, rows, differences


def summary_rows(name, plan, rows):
    fs = plan["sampling_rate_hz"]
    num_folds = len(plan["splits"])
    for count in plan["window_samples"]:
        selected = [row for row in rows if row["samples"] == count]
        acc, rate = "—", "—"
        if len(selected) == num_folds:
            accuracies = [row["acc_percent"] for row in selected]
            rates = [row["itr_bits_min"] for row in selected]
            acc = f"{np.mean(accuracies):.2f} ± {np.std(accuracies):.2f}"
            rate = f"{np.mean(rates):.2f} ± {np.std(rates):.2f}"
        yield f"| [{name}]({name}/report.md) | {count / fs:g} s | {len(selected)}/{num_folds} | {acc} | {rate} |"


def write_comparison(run_dir, all_results):
    header = ["| 模型 | 窗长 | 已评估折数 | ACC（%） | ITR（bits/min） |", "| --- | ---: | ---: | ---: | ---: |"]
    lines = ["# 留出被试测试汇总", "",
             "每个模型、窗长按被试汇总均值 ± 总体标准差。计划折数全部评估后显示汇总值。", "", *header]
    different = []
    for name, plan, rows, differences in all_results:
        if differences:
            different.append((name, plan, rows, differences))
        else:
            lines.extend(summary_rows(name, plan, rows))
    for name, plan, rows, differences in different:
        lines.extend(("", "## 旧实验", "", "口径差异为" + "、".join(differences) + "。", "", *header))
        lines.extend(summary_rows(name, plan, rows))
    if any(rows for _, _, rows, _ in all_results):
        lines.extend(("", "## 逐被试评估结果", "",
                      "| 模型 | 窗长（采样点） | 被试 | ACC（%） | ITR（bits/min） |",
                      "| --- | ---: | --- | ---: | ---: |"))
        for name, _, rows, _ in all_results:
            for row in rows:
                lines.append(f"| {name} | {row['samples']} | S{row['subject_id']:02d} | "
                             f"{row['acc_percent']:.2f} | {row['itr_bits_min']:.2f} |")
    save_text(run_dir / "report.md", "\n".join(lines) + "\n")


def evaluate_experiment(data_file, run_dir, legacy_run=None, legacy_data_file=None, preprocess_id=None):
    _, available_ids = configure_runtime()
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    path = artifacts / "run.json"
    with FileLock(artifacts / "run.lock", timeout=0), data_file.open("rb") as source:
        archive, digest = read_archive(source)
        with archive:
            method = archive_method(archive, preprocess_id)
            results = []
            if path.exists():
                plan = json.loads(path.read_text(encoding="utf-8"))
                if digest != plan["data_sha256"] or plan.get("preprocess_id", 1) != method:
                    raise ValueError("预处理数据的指纹与训练记录不符，请使用训练时的文件")
                assignments = model_devices(plan["model_order"], available_ids)
                for name in plan["model_order"]:
                    model_plan = {**plan, **plan.get("model_runs", {}).get(name, {})}
                    model_dir = run_dir / name
                    device, cuda_ids = configure_runtime(assignments[name])
                    rows = evaluate_model(name, archive, model_plan, model_dir, model_dir, device, cuda_ids)
                    results.append((name, model_plan, rows, []))
            elif legacy_run is None:
                raise FileNotFoundError(f"找不到训练记录 {path}")

            if legacy_run is not None:
                eeg_config = model_config("eeg-deformer")
                device, cuda_ids = configure_runtime(available_ids[:1])
                reference = next((entry[1] for entry in results if entry[0] == "eeg-deformer"), None)
                if reference is None:
                    counts = window_samples(data_config.WINDOW_SECONDS)
                    check_archive(archive, counts)
                    reference = {
                        "protocol": "loso_final_epoch", "settings": training_settings(eeg_config),
                        "window_samples": counts, "num_classes": data_config.NUM_CLASSES,
                        "runtime": runtime_details(device, cuda_ids), "models": {"eeg-deformer": eeg_config.SETTINGS},
                        "splits": loso_splits(archive["subject_ids"], config.LOSO_FOLDS),
                    }
                results.append(evaluate_legacy(run_dir, Path(legacy_run), legacy_data_file,
                                               archive, reference, device, cuda_ids))
            write_comparison(run_dir, results)
    print(f"测试汇总已保存到 {run_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    _, directory = experiment_args()
    eeg_config = model_config("eeg-deformer")
    for method, data_file, run_dir in config.experiment_datasets(directory):
        evaluate_experiment(data_file, run_dir, eeg_config.LEGACY_EEG_RUN_DIR,
                            eeg_config.LEGACY_EEG_DATA_FILE, method)
