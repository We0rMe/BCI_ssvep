import hashlib
import json
import os
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import evaluate
import train
from configs import config
from data_preprocess import config as data_config
from data_preprocess.preprocess import prepare_benchmark, save_archive


MODEL_NAMES = ("ssvepformer", "fb-ssvepformer", "spiketfnet", "eeg-deformer", "mtsnet")


def make_small_data(path):
    prepare_benchmark(data_config.BENCHMARK_DIR, path, (1, 2, 3), (0.5, 1.0))
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    for count in (125, 250):
        for kind in ("raw", "eeg"):
            key = f"{kind}_{count}"
            assert values[key].shape == (3, 240, 9, count)
            values[key] = values[key][:, :4].copy()
    for key in ("labels", "block_ids"):
        values[key] = values[key][:4]
    save_archive(path, values)


def check_mtsnet(data_file):
    module = train.model_module("mtsnet")
    settings = import_module("configs.mtsnet_config").SETTINGS
    with np.load(data_file, allow_pickle=False) as archive:
        for count in (125, 250):
            eeg, spectrum = train.prepare_inputs(module, archive, count, settings)
            np.testing.assert_array_equal(eeg, archive[f"eeg_{count}"])
            assert spectrum.shape == (*eeg.shape[:-1], 560)
            reference = np.fft.rfft(eeg[1, 0], n=1250)[:, 40:320] * (2 / count)
            np.testing.assert_allclose(spectrum[1, 0], np.concatenate((reference.real, reference.imag), axis=-1),
                                       rtol=1e-5, atol=1e-6)
            model = module.build_model(train.input_shapes((eeg, spectrum)), 40, settings)
            # 用 S2 的少量试次检查每条分支都能收到有限梯度。
            logits = model(torch.from_numpy(eeg[1]), torch.from_numpy(spectrum[1]))
            nn.functional.cross_entropy(logits, torch.from_numpy(archive["labels"])).backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def resume_fb(data_file, reference, interrupted):
    save_checkpoint = train.save_checkpoint

    def interrupt(path, state):
        if state["stage_index"] == 1:
            raise KeyboardInterrupt
        save_checkpoint(path, state)

    with patch.object(config, "MODELS", ("fb-ssvepformer",)), patch.object(data_config, "WINDOW_SECONDS", (0.5,)):
        with patch.object(train, "save_checkpoint", interrupt):
            try:
                train.train_experiment(data_file, interrupted)
            except KeyboardInterrupt:
                pass
            else:
                raise AssertionError("未触发阶段切换时的模拟中断")
        relative = Path("fb-ssvepformer/125pt_s01/artifacts")
        cp = interrupted / relative / "checkpoint.pt"
        assert torch.load(cp, weights_only=True)["stage_index"] == 0
        assert len(train.load_history(interrupted / relative / "training.log", 2)) == 2
        train.train_experiment(data_file, interrupted)

    first = torch.load(reference / relative / "checkpoint.pt", weights_only=True)
    resumed = torch.load(cp, weights_only=True)
    assert resumed["complete"] and resumed["epoch"] == 4 and resumed["best_epoch"] == first["best_epoch"]
    for key in ("model", "best_model", "optimizer", "scheduler", "torch_rng", "shuffle_rng"):
        torch.testing.assert_close(resumed[key], first[key], rtol=0, atol=0)
    assert len((interrupted / relative / "training.log").read_text(encoding="utf-8").splitlines()) == 5


def check_legacy(data_file, run_dir, legacy_dir, legacy_file):
    shared = json.loads((run_dir / "artifacts" / "run.json").read_text(encoding="utf-8"))
    with np.load(data_file, allow_pickle=False) as archive:
        save_archive(legacy_file, {key: archive[key] for key in archive.files if not key.startswith("raw_")})
    with legacy_file.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()

    mapping = evaluate.LEGACY_MODEL_KEYS
    options = shared["models"]["eeg-deformer"]
    old_settings = {**shared["settings"], **{key: options[name] for name, key in mapping.items()}, "NUM_CLASSES": 40}
    old_manifest = {"data_sha256": digest, "sampling_rate_hz": shared["sampling_rate_hz"],
                    "window_samples": shared["window_samples"], "runtime": shared["runtime"],
                    "splits": shared["splits"], "settings": old_settings}
    if "input" in options:
        old_manifest["input"] = options["input"]
    (legacy_dir / "artifacts").mkdir(parents=True)
    train.save_text(legacy_dir / "artifacts" / "run.json", json.dumps(old_manifest))
    for count in (125, 250):
        relative = Path(f"{count}pt_s01/artifacts/checkpoint.pt")
        state = torch.load(run_dir / "eeg-deformer" / relative, weights_only=True)
        destination = legacy_dir / relative
        destination.parent.mkdir(parents=True)
        torch.save({"epoch": state["stage_epoch"], "best_model": state["best_model"]}, destination)
    evaluate.evaluate_experiment(data_file, run_dir, legacy_dir, legacy_file)

    with np.load(run_dir / "eeg-deformer/artifacts/predictions.npz") as current, np.load(
        run_dir / "eeg-deformer-legacy/artifacts/predictions.npz"
    ) as historical:
        for count in (125, 250):
            np.testing.assert_array_equal(historical[f"logits_{count}"], current[f"logits_{count}"])
    provenance = json.loads((run_dir / "eeg-deformer-legacy/artifacts/source.json").read_text(encoding="utf-8"))
    assert provenance["protocol_differences"] == []


def check_extension(data_file, run_dir):
    model = ModuleType("models.example")
    model.prepare_inputs = lambda archive, count, settings: (archive[f"eeg_{count}"],)
    model.build_model = lambda shapes, classes, settings: nn.Sequential(nn.Flatten(), nn.Linear(np.prod(shapes[0]), classes))
    options = ModuleType("configs.example_config")
    options.SETTINGS = {}
    with (patch.dict("sys.modules", {model.__name__: model, options.__name__: options}),
          patch.object(config, "MODELS", ("example",)), patch.object(data_config, "WINDOW_SECONDS", (0.5,))):
        train.train_experiment(data_file, run_dir)
        evaluate.evaluate_experiment(data_file, run_dir)
    assert (run_dir / "example/artifacts/predictions.npz").is_file()


def record_worker(gpu_id, names, _data_file, run_dir, _manifest):
    (run_dir / f"gpu_{gpu_id}.json").write_text(json.dumps({"pid": os.getpid(), "models": names}), encoding="utf-8")


def check_two_gpu_schedule(root):
    assignments = train.model_devices(MODEL_NAMES, [0, 1])
    assert assignments == dict(zip(MODEL_NAMES, ([0], [1], [0], [1], [0]), strict=True))
    assert train.model_devices(("eeg-deformer",), [0, 1]) == {"eeg-deformer": [0, 1]}
    with patch.object(train, "train_on_gpu", record_worker):
        train.train_on_two_gpus(root / "benchmark.npz", root, {"model_devices": assignments}, [0, 1])
    first, second = [json.loads((root / f"gpu_{index}.json").read_text(encoding="utf-8")) for index in (0, 1)]
    assert first["pid"] != second["pid"]
    assert first["models"] == list(MODEL_NAMES[::2]) and second["models"] == list(MODEL_NAMES[1::2])


def main():
    output = data_config.PROJECT_DIR / "smoke_test" / "output"
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    fb = import_module("configs.fb-ssvepformer_config")
    with TemporaryDirectory(dir=output) as directory:
        root = Path(directory)
        check_two_gpu_schedule(root)
        data_file = root / "benchmark.npz"
        run_dir = root / "comparison"
        make_small_data(data_file)
        check_mtsnet(data_file)
        with (patch.object(data_config, "NUM_SUBJECTS", 3),
              patch.multiple(config, MODELS=MODEL_NAMES, DEVICE="cpu", BATCH_SIZE=4, EPOCHS=1, LOSO_FOLDS=1),
              patch.dict(fb.SETTINGS, {"pretrain_epochs": 1, "joint_epochs": 1})):
            train.train_experiment(data_file, run_dir)
            resume_fb(data_file, run_dir, root / "interrupted")
            check_legacy(data_file, run_dir, root / "legacy", root / "legacy.npz")
            check_extension(data_file, root / "extension")
        for name in MODEL_NAMES:
            for count in (125, 250):
                assert (run_dir / name / f"{count}pt_s01/train.png").read_bytes().startswith(b"\x89PNG")
            with np.load(run_dir / name / "artifacts" / "predictions.npz") as predictions:
                assert predictions["logits_125"].shape == predictions["logits_250"].shape == (1, 4, 40)

    summary = ("共享 SSVEP 流程 CPU 冒烟通过\n"
               "S1–S3 的真实试次，五模型、两个窗长，训练和留出评估通过。\n"
               "MTSNet 两个窗长的复频谱数值与参数梯度检查通过。\n"
               "FB 阶段切换中断恢复后与连续训练一致，旧 EEG checkpoint 格式读取与预测通过。\n"
               "接入测试模型无需修改训练或评估入口。已用两个独立进程验证双卡任务分配。\n")
    (output / "pipeline_report.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")


if __name__ == "__main__":
    main()
