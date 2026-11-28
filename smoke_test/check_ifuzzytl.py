"""用 Benchmark 固定窗检查模糊规则、真实 batch 和训练断点。"""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

import evaluate
import train
from configs import config, ifuzzytl_config as cfg
from data_preprocess import config as data_config
from models import ifuzzytl
from smoke_test.check_pipeline import make_small_data


@torch.no_grad()
def check_filter(layer, x):
    # 逐条规则计算后再加权，与模型中的批量投影独立对照。
    scores, values = [], []
    size = x.size(-1)
    for rule in range(layer.rules):
        rows = slice(rule * size, (rule + 1) * size)
        q = F.linear(x, layer.query.weight[rows], layer.query.bias[rows])
        diff = (q - layer.centers[rule]) / layer.log_width[rule].exp()
        scores.append(-0.5 * diff.square().sum(-1))
        values.append(F.linear(x, layer.value.weight[rows], layer.value.bias[rows]))
    weights = torch.stack(scores, dim=-1).softmax(-1)
    expected = sum(weights[..., i:i + 1] * value for i, value in enumerate(values))
    torch.testing.assert_close(layer.rule_weights(x), weights, atol=2e-5, rtol=1e-5)
    torch.testing.assert_close(layer(x), expected, atol=2e-5, rtol=1e-5)


def check_batch(archive, count):
    inputs = train.prepare_inputs(ifuzzytl, archive, count, cfg.SETTINGS)
    x = torch.from_numpy(inputs[0][1, :cfg.BATCH_SIZE])  # S2；S1 留出
    y = torch.from_numpy(archive["labels"][:len(x)])
    model = ifuzzytl.build_model(train.input_shapes(inputs), 40, cfg.SETTINGS)
    check_filter(model.spatial, x[:2])
    check_filter(model.temporal, model.spatial(x[:2]).detach().transpose(1, 2))
    logits = model(x)
    assert logits.shape == (len(y), 40) and logits.dtype == torch.float32
    loss = F.cross_entropy(logits, y)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max() > 0, name
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(x[:1]), model(x[:3])[:1], atol=2e-5, rtol=1e-5)
    return f"{count} 点：真实输入 {list(x.shape)}，FP32 logits {list(logits.shape)}，初始交叉熵 {loss.item():.4f}。"


def check_resume(data_file, directory):
    reference, resumed = directory / "reference", directory / "resumed"
    save_checkpoint = train.save_checkpoint

    def stop_after_first(path, state):
        save_checkpoint(path, state)
        raise InterruptedError("模拟中断")

    with (patch.object(data_config, "NUM_SUBJECTS", 3),
          patch.multiple(config, DEVICE="cpu", LOSO_FOLDS=1),
          patch.multiple(cfg, BATCH_SIZE=4, EPOCHS=3, WARMUP_EPOCHS=2)):
        train.train_experiment(data_file, reference, ("ifuzzytl",))
        with patch.object(train, "save_checkpoint", stop_after_first):
            try:
                train.train_experiment(data_file, resumed, ("ifuzzytl",))
            except InterruptedError:
                pass
        train.train_experiment(data_file, resumed, ("ifuzzytl",))
        evaluate.evaluate_experiment(data_file, resumed)
    manifest = json.loads((resumed / "artifacts/run.json").read_text(encoding="utf-8"))
    assert manifest["model_runs"]["ifuzzytl"]["splits"] == [{"test": 1, "train": [2, 3]}]
    with np.load(resumed / "ifuzzytl/artifacts/predictions.npz") as predictions:
        for count in (125, 250):
            artifacts = Path("ifuzzytl") / f"{count}pt_s01" / "artifacts"
            state = torch.load(resumed / artifacts / "checkpoint.pt", weights_only=True)
            baseline = torch.load(reference / artifacts / "checkpoint.pt", weights_only=True)
            assert state["complete"] and state["epoch"] == 3
            for name, value in baseline["model"].items():
                torch.testing.assert_close(state["model"][name], value, rtol=0, atol=0)
            with (resumed / artifacts / "training.log").open(encoding="utf-8") as log:
                rows = list(csv.DictReader(log))
            rates = [float(row["lr"]) for row in rows]
            np.testing.assert_allclose(rates, np.array([0.5, 1, 0.5]) * cfg.LEARNING_RATE)
            assert all(float(row["train_seconds"]) > 0 for row in rows)
            np.testing.assert_array_equal(predictions[f"subject_ids_{count}"], [1])
            logits = predictions[f"logits_{count}"]
            assert logits.shape == (1, 4, 40) and np.isfinite(logits).all()


def main():
    torch.set_num_threads(2)
    torch.manual_seed(config.SEED)
    output = data_config.PROJECT_DIR / "smoke_test/output/ifuzzytl"
    output.mkdir(parents=True, exist_ok=True)
    with np.load(data_config.OUTPUT_FILES[1]) as archive:
        train.check_archive(archive, (125, 250))
        summary = [check_batch(archive, count) for count in (125, 250)]
    with TemporaryDirectory(dir=output) as tmp:
        directory = Path(tmp)
        data_file = directory / "benchmark.npz"
        make_small_data(data_file)
        with patch.object(config, "WINDOW_MODE", "fixed"):
            check_resume(data_file, directory)
    summary += ["CPU 冒烟通过。两个窗长的规则输出、梯度和逐试次推理通过核对。",
                "微型 LOSO 每人 4 个真实试次，S2、S3 训练，S1 评估。预热、衰减、训练耗时和断点恢复已验证。"]
    report = "\n".join(summary) + "\n"
    (output / "cpu.txt").write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
