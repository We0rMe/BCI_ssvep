import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock

import evaluate
import train
from data_preprocess import config as data_config
from models import ssvepgpt
from . import readout


SOURCE = data_config.PROJECT_DIR / "results" / "ssvepgpt_rms"
OUTPUT = data_config.PROJECT_DIR / "results" / "ssvepgpt_readout"
SETTINGS = dict(SEED=42, BATCH_SIZE=64, EPOCHS=100, OPTIMIZER="adam",
                LEARNING_RATE=0.001, WEIGHT_DECAY=1e-5, SCHEDULER="cosine")


@dataclass
class FoldResult:
    train_acc: float
    train_seconds: float
    logits: np.ndarray
    row: dict


def checkpoint(directory):
    return directory / "artifacts" / "checkpoint.pt"


def source_plan(source_dir, archive, digest, folds):
    path = source_dir / "artifacts" / "run.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["data_sha256"] != digest or manifest["protocol"] != "loso_final_epoch":
        raise ValueError("源实验需要相同的预处理文件和固定末轮 LOSO 协议")
    options = manifest["models"]["ssvepgpt"]
    if not options.get("normalize_trial_rms") or options.get("spectrum_bands") != 4:
        raise ValueError("源实验需要使用逐试次 RMS 和四频带 SsvepGPT")
    run = manifest["model_runs"]["ssvepgpt"]
    splits = run["splits"]
    if splits != train.loso_splits(archive["subject_ids"], len(splits)):
        raise ValueError("源实验划分与按被试编号留一的完整训练集不符")
    if folds is not None and not 1 <= folds <= len(splits):
        raise ValueError("--folds 超出源实验折数")
    return manifest, splits[:folds]


def load_backbone(path, shapes, manifest, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    epochs = manifest["model_runs"]["ssvepgpt"]["settings"]["EPOCHS"]
    if not state["complete"] or state["epoch"] != epochs:
        raise ValueError(f"源主干尚未完成固定轮数训练：{path}")
    model = ssvepgpt.build_model(shapes, manifest["num_classes"], manifest["models"]["ssvepgpt"])
    model.load_state_dict(state["model"])
    return model.to(device).eval().requires_grad_(False)


def run_window(count, archive, manifest, splits, source_dir, output, settings, device, ids, results):
    inputs = train.prepare_inputs(ssvepgpt, archive, count, manifest["models"]["ssvepgpt"])
    subjects, labels = archive["subject_ids"], archive["labels"]
    classes, fs = manifest["num_classes"], manifest["sampling_rate_hz"]
    batch = settings["BATCH_SIZE"]
    gaze = manifest["model_runs"]["ssvepgpt"]["settings"]["ITR_GAZE_SECONDS"]
    plan = [("main", settings["EPOCHS"], None)]
    for split in splits:
        subject = split["test"]
        fold = train.fold_directory(source_dir / "ssvepgpt", count, subject)
        backbone = load_backbone(checkpoint(fold), train.input_shapes(inputs), manifest, device)
        features = (readout.extract(backbone, inputs, batch, device),)
        for arm in readout.ARMS:
            folder = train.fold_directory(output / arm, count, subject)
            options = {"arm": arm}
            acc, seconds = train.train_fold(readout, options, plan, settings, features, labels,
                                            subjects, split, folder, device, ids)
            model = readout.build_model(train.input_shapes(features), classes, options).to(device)
            state = torch.load(checkpoint(folder), map_location="cpu", weights_only=True)
            model.load_state_dict(state["model"])
            loader = train.subject_loader(features, labels, subjects, [subject], batch, device)
            logits = evaluate.predict(model, loader, device)
            correct = int((logits.argmax(1) == labels).sum())
            accuracy = correct / len(labels)
            row = {
                "samples": count, "subject_id": subject, "seed": settings["SEED"],
                "correct": correct, "trials": len(labels), "acc_percent": 100 * accuracy,
                "itr_bits_min": evaluate.itr(accuracy, classes, count / fs + gaze),
            }
            results[arm].append(FoldResult(acc, seconds, logits, row))
        del features


def write_reports(output, archive, manifest, splits, results):
    counts, fs = manifest["window_samples"], manifest["sampling_rate_hz"]
    summary = []
    for arm, entries in results.items():
        predictions = {"labels": archive["labels"], "block_ids": archive["block_ids"]}
        training = []
        for count in counts:
            group = [entry for entry in entries if entry.row["samples"] == count]
            training.append((count, [e.train_acc for e in group], sum(e.train_seconds for e in group)))
            predictions[f"subject_ids_{count}"] = np.array([e.row["subject_id"] for e in group])
            predictions[f"logits_{count}"] = np.stack([e.logits for e in group])
        rows = [entry.row for entry in entries]
        train.write_training_report(output / arm, arm, fs, training)
        evaluate.write_test_details(output / arm, arm, rows, predictions)
        summary.append((arm, dict(sampling_rate_hz=fs, window_samples=counts, splits=splits), rows, []))
    evaluate.write_comparison(output, summary)
    report = output / "report.md"
    lines = ["", "## 同一被试的 B − A", "", "| 窗长 | 平均 ACC 差（百分点） | B 胜 / 平 / 负 |",
             "| --- | ---: | ---: |"]
    for count in counts:
        a, b = [[e.row["correct"] for e in results[arm] if e.row["samples"] == count]
                for arm in readout.ARMS]
        diff = np.subtract(b, a)
        delta = 100 * diff.mean() / len(archive["labels"])
        lines.append(f"| {count / fs:g} s | {delta:+.2f} | "
                     f"{sum(diff > 0)} / {sum(diff == 0)} / {sum(diff < 0)} |")
    train.save_text(report, report.read_text(encoding="utf-8") + "\n".join(lines) + "\n")


def run(data_file, source_dir, output, folds, settings, device, ids):
    source_dir, output = source_dir.resolve(), output.resolve()
    if output.is_relative_to(source_dir) or source_dir.is_relative_to(output):
        raise ValueError("请为读出实验指定独立输出目录")
    run_file = output / "artifacts" / "readout.json"
    lock_file = run_file.with_suffix(".lock")
    if output.exists() and not run_file.exists() and any(
        path not in (run_file.parent, lock_file) for path in output.rglob("*")
    ):
        raise ValueError("该目录已有其他内容，请指定新的读出实验目录")
    run_file.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(lock_file, timeout=0), data_file.open("rb") as file:
        archive, digest = train.read_archive(file)
        with archive:
            manifest, splits = source_plan(source_dir, archive, digest, folds)
            counts = manifest["window_samples"]
            train.check_archive(archive, counts)
            if manifest["num_classes"] != data_config.NUM_CLASSES:
                raise ValueError("源模型类别数与训练接口不一致")
            hashes = {}
            for count in counts:
                for split in splits:
                    folder = train.fold_directory(source_dir / "ssvepgpt", count, split["test"])
                    with checkpoint(folder).open("rb") as weights:
                        hashes[folder.name] = hashlib.file_digest(weights, "sha256").hexdigest()
            train.save_manifest(run_file, {
                "protocol": "frozen_rms_post_norm_readout_v1", "arms": readout.ARMS,
                "settings": settings, "source_manifest": manifest, "splits": splits,
                "source_checkpoint_sha256": hashes, "runtime": train.runtime_details(device, ids),
            })
            results = {arm: [] for arm in readout.ARMS}
            for count in counts:
                run_window(count, archive, manifest, splits, source_dir, output, settings, device, ids, results)
            write_reports(output, archive, manifest, splits, results)


def main():
    parser = argparse.ArgumentParser(description="配对比较冻结 RMS 主干的 A/B 线性读出")
    parser.add_argument("--source-run", type=Path, default=SOURCE)
    parser.add_argument("--data-file", type=Path, default=data_config.OUTPUT_FILE)
    parser.add_argument("--run-dir", type=Path, default=OUTPUT)
    parser.add_argument("--folds", type=int, help="从 S01 开始，默认使用源实验全部折")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA 模式下使用的单卡编号")
    parser.add_argument("--cpu", action="store_true", help="临时使用 CPU")
    args = parser.parse_args()
    if args.cpu:
        torch.use_deterministic_algorithms(True)
        device, ids = torch.device("cpu"), []
    else:
        device, ids = train.configure_runtime([args.gpu])
    run(args.data_file, args.source_run, args.run_dir, args.folds, SETTINGS, device, ids)
    print(f"读出比较已写入 {args.run_dir / 'report.md'}")


if __name__ == "__main__":
    main()
