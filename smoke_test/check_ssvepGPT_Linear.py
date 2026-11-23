from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

import evaluate
import train
from configs import config, ssvepGPT_Linear_config
from data_preprocess import config as data_config
from smoke_test.check_pipeline import make_small_data


NAME = "ssvepGPT_Linear"


def check_window(archive, count):
    module = train.model_module(NAME)
    settings = ssvepGPT_Linear_config.SETTINGS
    inputs = train.prepare_inputs(module, archive, count, settings)
    (spectrum,) = inputs
    eeg = archive[f"eeg_{count}"].astype(np.float64)
    rms = np.sqrt(np.mean(eeg ** 2, axis=(-2, -1), keepdims=True))
    normalized = (eeg / np.maximum(rms, 1e-12)).astype(np.float32)
    fft = np.fft.rfft(normalized, n=1250, axis=-1)[..., 40:321] / count
    np.testing.assert_allclose(spectrum, np.concatenate((fft.real, fft.imag), axis=-1),
                               rtol=1e-5, atol=1e-6)

    model = module.build_model(train.input_shapes(inputs), 40, settings)
    x = torch.from_numpy(spectrum[1])
    y = torch.from_numpy(archive["labels"])
    w, b = model.parameters()
    torch.testing.assert_close(model(x), x.flatten(1) @ w.T + b)

    # 核对第一步 Adam 的一阶矩，确认 L2 项实际进入更新。
    optimizer = train.make_optimizer(model, train.training_settings(ssvepGPT_Linear_config))
    initial = [p.detach().clone() for p in (w, b)]
    loss, _, _ = train.batch_step(model, (x, y), torch.device("cpu"), optimizer)
    assert torch.isfinite(loss) and ssvepGPT_Linear_config.WEIGHT_DECAY > 0
    beta1 = optimizer.param_groups[0]["betas"][0]
    for p, before in zip((w, b), initial):
        expected = (1 - beta1) * (p.grad + ssvepGPT_Linear_config.WEIGHT_DECAY * before)
        torch.testing.assert_close(optimizer.state[p]["exp_avg"], expected, atol=1e-10, rtol=1e-5)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(config.SEED)
    output = data_config.PROJECT_DIR / "smoke_test" / "output"
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output) as directory:
        data_file, run_dir = Path(directory) / "benchmark.npz", Path(directory) / "run"
        make_small_data(data_file)
        with np.load(data_file) as archive:
            for count in (125, 250):
                check_window(archive, count)
        with (patch.object(data_config, "NUM_SUBJECTS", 3),
              patch.multiple(config, DEVICE="cpu", LOSO_FOLDS=1),
              patch.multiple(ssvepGPT_Linear_config, EPOCHS=1, BATCH_SIZE=4)):
            train.train_experiment(data_file, run_dir, (NAME,))
            evaluate.evaluate_experiment(data_file, run_dir)
        with np.load(run_dir / NAME / "artifacts" / "predictions.npz") as predictions:
            for count in (125, 250):
                np.testing.assert_array_equal(predictions[f"subject_ids_{count}"], [1])
                logits = predictions[f"logits_{count}"]
                assert logits.shape == (1, 4, 40) and np.isfinite(logits).all()

    report = f"{NAME} CPU 冒烟通过。两个真实窗长的 RMS 复谱、线性映射、L2 更新及末轮权重评估均已核对。\n"
    (output / f"{NAME}.txt").write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
