"""在相同的开发被试上比较原版、固定小波和可学习小波。"""

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from filelock import FileLock

import evaluate
import train
from configs import config, mtsnet_config
from data_preprocess import config as data_config
from data_preprocess.preprocess import window_samples
from models import mtsnet
from smoke_test.check_gpu import probe
from . import arms


ARMS = ("baseline", "fixed", "learned")
# S1 留作外层被试。开发被试编号预先固定，换训练种子时也保持一致。
OUTER_SUBJECT = 1
DEV_SUBJECTS = (4, 9, 14, 19, 24, 29)


def run_arm(name, gpu, seed, preflight):
    device, ids = train.configure_runtime([gpu] if gpu is not None else [])
    module = mtsnet if name == "baseline" else arms
    options = dict(mtsnet_config.SETTINGS, arm=name, wavelet_penalty=1.0)
    recipe = dict(train.training_settings(mtsnet_config), SEED=seed)
    epochs = recipe["EPOCHS"]
    counts = window_samples(data_config.WINDOW_SECONDS)
    source = data_config.OUTPUT_FILES[1]
    root = data_config.PROJECT_DIR / ("smoke_test/output" if preflight else "results")
    folder = root / "mtsnet_wavelet_dev" / f"seed{seed}" / name
    folder.mkdir(parents=True, exist_ok=True)
    runtime = train.runtime_details(device, ids)
    with FileLock(folder / "run.lock", timeout=0), source.open("rb") as file:
        archive, digest = train.read_archive(file)
        with archive:
            train.archive_method(archive, 1)
            train.check_archive(archive, counts)
            subjects, labels = archive["subject_ids"], archive["labels"]
            training = sorted(set(subjects.tolist()) - {OUTER_SUBJECT, *DEV_SUBJECTS})
            split = {"train": training}
            train.save_manifest(folder / "run.json", dict(
                protocol="mtsnet_wavelet_inner_dev_v1", data_sha256=digest, runtime=runtime,
                train=training, dev=DEV_SUBJECTS, outer=OUTER_SUBJECT, windows=counts,
                options=options, recipe=recipe,
            ))
            rows, checks, filters = [], {}, []
            for count in counts:
                inputs = train.prepare_inputs(mtsnet, archive, count, options)
                if preflight:
                    checks[str(count)] = probe(module, options, inputs, labels, subjects, split,
                                               device, ids, recipe)
                    continue
                work = folder / f"{count}pt"
                train.train_fold(module, options, [("main", epochs, None)], recipe, inputs, labels,
                                 subjects, split, work, device, ids)
                model = module.build_model(train.input_shapes(inputs), data_config.NUM_CLASSES, options).to(device)
                if name != "baseline":
                    initial = model.wavelet.low.detach().clone()
                state = torch.load(work / "artifacts/checkpoint.pt", map_location="cpu", weights_only=True)
                model.load_state_dict(state["model"])
                del state
                if name != "baseline":
                    change = (model.wavelet.low.detach() - initial).norm() / initial.norm()
                    filters.append((count, *model.gains.detach().tolist(), 100 * change.item()))
                for subject in DEV_SUBJECTS:
                    loader = train.subject_loader(inputs, labels, subjects, [subject], recipe["BATCH_SIZE"], device)
                    logits = evaluate.predict(model, loader, device)
                    rows.append((count, subject, int((logits.argmax(1) == labels).sum()), len(labels)))
                del model, inputs
            if preflight:
                train.save_text(folder / "gpu_report.json", json.dumps(dict(runtime=runtime, windows=checks),
                                                                      indent=2, ensure_ascii=False))
            return rows, filters


def run_group(names, gpu, seed, preflight):
    return {name: run_arm(name, gpu, seed, preflight) for name in names}


def save_report(results, seed):
    folder = data_config.PROJECT_DIR / "results/mtsnet_wavelet_dev" / f"seed{seed}"
    lines = ["# MTSNet 小波开发集对照", "", "S1 外层留出。各组在同一批开发被试上评估末轮权重。", "",
             "| 窗长（点） | 被试 | 原版 ACC（%） | 固定 ACC（%） | 可学习 ACC（%） | 固定 − 原版（pp） | 可学习 − 固定（pp） |",
             "| ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    scores = {name: {(n, s): 100 * k / total for n, s, k, total in rows}
              for name, (rows, _) in results.items()}
    for count in sorted({n for n, _ in scores["baseline"]}):
        values = np.array([[scores[name][count, s] for name in ARMS] for s in DEV_SUBJECTS])
        for label, (a, b, c) in zip([*DEV_SUBJECTS, "平均"], [*values, values.mean(axis=0)], strict=True):
            lines.append(f"| {count} | {label} | {a:.2f} | {b:.2f} | {c:.2f} | {b-a:+.2f} | {c-b:+.2f} |")
    lines.extend(["", "## 小波支路末轮参数", "",
                  "| 组别 | 窗长（点） | A3 权重 | D3 权重 | D2 权重 | 小波核相对 db4 的 L2 变化（%） |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for name in ("fixed", "learned"):
        for n, a3, d3, d2, change in results[name][1]:
            lines.append(f"| {name} | {n} | {a3:.5f} | {d3:.5f} | {d2:.5f} | {change:.4f} |")
    train.save_text(folder / "report.md", "\n".join(lines) + "\n")
    print(folder / "report.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--gpus", type=int, nargs="+", default=config.CUDA_DEVICE_IDS)
    parser.add_argument("--probe", action="store_true", help="只用真实训练 batch 检查设备、反向和耗时")
    args = parser.parse_args()
    _, ids = train.configure_runtime(args.gpus)
    if args.probe and not ids:
        parser.error("GPU 预检需要 DEVICE='cuda'")
    if len(ids) == 2:
        with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context("spawn")) as pool:
            tasks = [pool.submit(run_group, ARMS[i::2], gpu, args.seed, args.probe) for i, gpu in enumerate(ids)]
            results = {name: rows for task in tasks for name, rows in task.result().items()}
    else:
        results = run_group(ARMS, ids[0] if ids else None, args.seed, args.probe)
    if args.probe:
        print(f"GPU 预检完成：smoke_test/output/mtsnet_wavelet_dev/seed{args.seed}/各组/gpu_report.json")
    else:
        save_report(results, args.seed)
