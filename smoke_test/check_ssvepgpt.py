from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

import evaluate
import train
from configs import config
from configs.ssvepgpt_config import SETTINGS
from data_preprocess import config as data_config
from smoke_test.check_pipeline import make_small_data


def check_window(archive, count):
    module = train.model_module("ssvepgpt")
    inputs = train.prepare_inputs(module, archive, count, SETTINGS)
    eeg, spectrum = inputs
    np.testing.assert_array_equal(eeg, archive[f"eeg_{count}"])
    expected = np.fft.rfft(eeg, n=1250, axis=-1)[..., 40:321] / count
    np.testing.assert_allclose(spectrum, np.concatenate((expected.real, expected.imag), axis=-1),
                               rtol=1e-5, atol=1e-6)
    model = module.build_model(train.input_shapes(inputs), 40, SETTINGS)
    batch = [torch.from_numpy(array[1]) for array in inputs]
    logits = model(*batch)
    assert logits.shape == (4, 40)
    torch.nn.functional.cross_entropy(logits, torch.from_numpy(archive["labels"])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    for weight in (model.time_projection.weight, model.spectrum_projection.weight,
                   model.blocks[0].attention.in_proj_weight):
        assert weight.grad.abs().max() > 0

    # 分别改变一个域的输入，检查第一层中另一个域的输出是否跟着改变。
    model.eval()
    outputs = []
    hook = model.blocks[0].register_forward_hook(lambda _, args, result: outputs.append(result))
    with torch.no_grad():
        model(*batch)
        model(torch.zeros_like(batch[0]), batch[1])
        model(batch[0], torch.zeros_like(batch[1]))
    hook.remove()
    time_count = eeg.shape[2] * (count // SETTINGS["patch_samples"])
    assert not torch.allclose(outputs[0][:, time_count:], outputs[1][:, time_count:])
    assert not torch.allclose(outputs[0][:, :time_count], outputs[2][:, :time_count])
    return sum(p.numel() for p in model.parameters())


def main():
    torch.set_num_threads(2)
    torch.manual_seed(config.SEED)
    output = data_config.PROJECT_DIR / "smoke_test" / "output"
    output.mkdir(parents=True, exist_ok=True)
    summary = [f"SsvepGPT CPU 冒烟，PyTorch {torch.__version__}，S1–S3 各 4 个真实试次。"]
    with TemporaryDirectory(dir=output) as temp:
        data_file, run_dir = Path(temp) / "benchmark.npz", Path(temp) / "run"
        make_small_data(data_file)
        with np.load(data_file) as archive:
            for count in (125, 250):
                parameters = check_window(archive, count)
                summary.append(f"{count} 点：{parameters:,} 个参数，复频谱、双向跨域交互及梯度检查通过。")
        with (patch.object(data_config, "NUM_SUBJECTS", 3),
              patch.multiple(config, MODELS=("ssvepgpt",), DEVICE="cpu", BATCH_SIZE=4, EPOCHS=1, LOSO_FOLDS=1)):
            train.train_experiment(data_file, run_dir)
            evaluate.evaluate_experiment(data_file, run_dir)
        with np.load(run_dir / "ssvepgpt" / "artifacts" / "predictions.npz") as predictions:
            for count in (125, 250):
                np.testing.assert_array_equal(predictions[f"subject_ids_{count}"], [1])
                logits = predictions[f"logits_{count}"]
                assert logits.shape == (1, 4, 40) and np.isfinite(logits).all()
    summary.append("共用训练、最佳权重重载及留出被试预测通过。")
    report = "\n".join(summary) + "\n"
    (output / "ssvepgpt.txt").write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
