"""用真实 RMS 权重检查两种读出，仅使用 S01 折中的训练被试。"""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

import train
from configs import config, ssvepgpt_config
from data_preprocess import config as data_config
from experiments.ssvepgpt_readout import readout
from experiments.ssvepgpt_readout.__main__ import SOURCE, SETTINGS, checkpoint, load_backbone, run, source_plan
from models import ssvepgpt
from smoke_test.check_gpu import probe


def check_small_run(data_file, output, device):
    # 三名被试、每人八个真实试次，检查训练、测试、续跑和结果汇总。
    with TemporaryDirectory(dir=output) as tmp:
        root = Path(tmp)
        small = root / "small.npz"
        source, target = root / "source", root / "readout"
        with np.load(data_file) as archive:
            values = {}
            for key in archive.files:
                value = archive[key]
                if key.startswith(("eeg_", "raw_")):
                    value = value[:3, :8].copy()
                elif key == "subject_ids":
                    value = value[:3]
                elif key in ("labels", "block_ids"):
                    value = value[:8]
                values[key] = value
            np.savez(small, **values)
        settings = dict(SETTINGS, EPOCHS=1, BATCH_SIZE=4)
        with patch.object(data_config, "NUM_SUBJECTS", 3), patch.multiple(config, DEVICE="cpu", LOSO_FOLDS=1), \
                patch.multiple(ssvepgpt_config, EPOCHS=1, BATCH_SIZE=4):
            train.train_experiment(small, source, ("ssvepgpt",))
            run(small, source, target, 1, settings, device, [])
            saved = []
            for arm in readout.ARMS:
                with np.load(target / arm / "artifacts" / "predictions.npz") as pred:
                    for count in (125, 250):
                        assert pred[f"subject_ids_{count}"].tolist() == [1]
                        assert pred[f"logits_{count}"].shape == (1, 8, 40)
                        saved.append(pred[f"logits_{count}"])
            run(small, source, target, 1, settings, device, [])
            for i, arm in enumerate(readout.ARMS):
                with np.load(target / arm / "artifacts" / "predictions.npz") as pred:
                    for j, count in enumerate((125, 250)):
                        np.testing.assert_array_equal(saved[2 * i + j], pred[f"logits_{count}"])
            assert "B − A" in (target / "report.md").read_text(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="在本地 CPU 上做冒烟测试")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--source-run", type=Path, default=SOURCE)
    parser.add_argument("--data-file", type=Path, default=data_config.OUTPUT_FILE)
    args = parser.parse_args()
    if args.cpu:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        device, ids = torch.device("cpu"), []
    else:
        device, ids = train.configure_runtime([args.gpu])
    report = {"runtime": train.runtime_details(device, ids), "windows": {},
              "warmup_batches": config.GPU_WARMUP_BATCHES, "timed_batches": config.GPU_PROBE_BATCHES}
    with args.data_file.open("rb") as file:
        archive, digest = train.read_archive(file)
        with archive:
            manifest, (split,) = source_plan(args.source_run, archive, digest, 1)
            batch = SETTINGS["BATCH_SIZE"]
            labels, subjects = archive["labels"], archive["subject_ids"]
            batches = config.GPU_WARMUP_BATCHES + config.GPU_PROBE_BATCHES + 1
            n = (batches * batch + len(labels) - 1) // len(labels)
            selected = split["train"][:n]
            indices = [subjects.tolist().index(s) for s in selected]
            for count in manifest["window_samples"]:
                key = f"eeg_{count}"
                subset = {key: archive[key][indices], "sampling_rate_hz": archive["sampling_rate_hz"]}
                inputs = ssvepgpt.prepare_inputs(subset, count, manifest["models"]["ssvepgpt"])
                fold = train.fold_directory(args.source_run / "ssvepgpt", count, split["test"])
                backbone = load_backbone(checkpoint(fold), train.input_shapes(inputs), manifest, device)
                features = readout.extract(backbone, inputs, batch, device)
                tokens = torch.from_numpy(features[0, :batch]).to(device)
                sample = [torch.from_numpy(x[0, :batch]).to(device) for x in inputs]
                with torch.inference_mode(), profile(activities=[ProfilerActivity.CPU], acc_events=True) as trace:
                    expected = backbone(*sample)
                    actual = backbone.classifier(readout.read_features(tokens, "A_mean"))
                    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
                    flat = readout.read_features(tokens, "B_flat")
                    torch.testing.assert_close(flat[:, tokens.size(-1):].reshape_as(tokens[:, 1:]), tokens[:, 1:])
                window = {"source_fold": fold.name, "feature_shape": list(tokens.shape),
                          "original_head_max_abs_error": (expected - actual).abs().max().item(),
                          "backbone_operators": sorted(e.key for e in trace.key_averages()
                                                       if "attention" in e.key or e.key == "aten::bmm")}
                for arm in readout.ARMS:
                    window[arm] = probe(readout, {"arm": arm}, (features,), labels, selected,
                                        {"train": selected}, device, ids, SETTINGS)
                assert all(not p.requires_grad and p.grad is None for p in backbone.parameters())
                report["windows"][str(count)] = window
    output = data_config.PROJECT_DIR / "smoke_test" / "output" / "ssvepgpt_readout"
    output.mkdir(parents=True, exist_ok=True)
    if args.cpu:
        check_small_run(args.data_file, output, device)
        report["small_run_and_resume_passed"] = True
    path = output / f"{device.type}_report.json"
    train.save_text(path, json.dumps(report, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
