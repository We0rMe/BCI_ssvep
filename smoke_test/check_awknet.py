from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

import evaluate
import train
from configs import awknet_config, config
from data_preprocess import config as data_config
from models import awknet
from smoke_test.check_pipeline import make_small_data


def check_window(archive, count):
    settings = awknet_config.SETTINGS
    inputs = train.prepare_inputs(awknet, archive, count, settings)
    np.testing.assert_array_equal(inputs[0], archive[f"eeg_{count}"])
    eeg = torch.from_numpy(inputs[0][1])  # S2，S1 只用于末轮评估
    labels = torch.from_numpy(archive["labels"])
    model = awknet.build_model(train.input_shapes(inputs), 40, settings)
    wavelet = model.wavelet
    a3, d3, d2 = wavelet(eeg)
    assert [band.shape[-1] for band in (a3, d3, d2)] == [(count + 7) // 8] * 2 + [(count + 3) // 4]
    assert wavelet.penalty().item() < 1e-10

    # 独立用逐点循环索引核对奇偶窗的延拓方向、下采样相位和细节系数。
    low = eeg.numpy()
    taps = wavelet.low.detach().numpy()
    high = taps[::-1] * np.array([1, -1] * 4)
    details = []
    for _ in range(3):
        if low.shape[-1] % 2:
            low = np.concatenate((low, low[..., -1:]), axis=-1)
        indices = (np.arange(0, low.shape[-1], 2)[:, None] + np.arange(8) - 3) % low.shape[-1]
        windows = low[..., indices]
        low = windows @ taps
        details.append(windows @ high)
    for band, reference in zip((a3, d3, d2), (low, details[2], details[1])):
        np.testing.assert_allclose(band.detach().numpy(), reference, atol=2e-5, rtol=1e-5)

    optimizer = train.make_optimizer(model, train.training_settings(awknet_config))
    assert [g["lr"] for g in optimizer.param_groups] == [awknet_config.LEARNING_RATE, settings["wavelet_lr"]]
    assert optimizer.param_groups[1]["weight_decay"] == 0
    initial = wavelet.low.detach().clone()
    output = model(eeg)
    assert output["logits"].shape == (len(labels), 40)
    loss = awknet.joint_loss(output, labels, settings)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert wavelet.low.grad.abs().max() > 0
    optimizer.step()
    assert not torch.equal(initial, wavelet.low)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(config.SEED)
    output_dir = data_config.PROJECT_DIR / "smoke_test" / "output" / "awknet"
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir) as directory:
        data_file = Path(directory) / "benchmark.npz"
        run_dir = Path(directory) / "run"
        make_small_data(data_file)
        with np.load(data_file) as archive:
            for count in (125, 250):
                check_window(archive, count)
        with (patch.object(data_config, "NUM_SUBJECTS", 3),
              patch.multiple(config, DEVICE="cpu", LOSO_FOLDS=1),
              patch.multiple(awknet_config, BATCH_SIZE=4, EPOCHS=1)):
            train.train_experiment(data_file, run_dir, ("awknet",))
            evaluate.evaluate_experiment(data_file, run_dir)
        with np.load(run_dir / "awknet" / "artifacts" / "predictions.npz") as predictions:
            for count in (125, 250):
                np.testing.assert_array_equal(predictions[f"subject_ids_{count}"], [1])
                logits = predictions[f"logits_{count}"]
                assert logits.shape == (1, 4, 40) and np.isfinite(logits).all()

    summary = ("AWKNet CPU 冒烟通过。S2、S3 训练，S1 留出，每人 4 个真实试次。\n"
               "125/250 点小波系数、分组学习率、小波更新、末轮权重加载和评估接口均通过。\n")
    (output_dir / "cpu.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")


if __name__ == "__main__":
    main()
