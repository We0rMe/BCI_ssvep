from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
from scipy.interpolate import BSpline

import evaluate
import train
from configs import config, ssvepformer_eKAN_config
from configs.ssvepformer_eKAN_config import SETTINGS
from data_preprocess import config as data_config
from smoke_test.check_pipeline import make_small_data


def main():
    name = "ssvepformer_eKAN"
    output = data_config.PROJECT_DIR / "smoke_test" / "output"
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    module = train.model_module(name)
    with TemporaryDirectory(dir=output) as directory:
        data_file = Path(directory) / "benchmark.npz"
        run_dir = Path(directory) / "run"
        make_small_data(data_file)
        with np.load(data_file) as archive:
            for count in (125, 250):
                inputs = train.prepare_inputs(module, archive, count, SETTINGS)
                assert inputs[0].shape == (3, 4, 9, 562)
                model = module.build_model(train.input_shapes(inputs), 40, SETTINGS)
                head = model.classifier[-1]
                # 用独立的 SciPy 实现核对网格内的基函数，包括两端点。
                x = torch.linspace(*SETTINGS["grid_range"], steps=51)
                expected = BSpline.design_matrix(x.numpy(), head.knots.numpy(), head.order).toarray()
                np.testing.assert_allclose(head.spline_basis(x).numpy(), expected, atol=1e-6)

                logits = model(torch.from_numpy(inputs[0][1]))
                assert logits.shape == (4, 40)
                torch.nn.functional.cross_entropy(logits, torch.from_numpy(archive["labels"])).backward()
                for weight in (head.base.weight, head.spline_weight, model.channel_combination[0].weight):
                    assert torch.isfinite(weight.grad).all() and weight.grad.abs().max() > 0

        with (patch.object(data_config, "NUM_SUBJECTS", 3),
              patch.multiple(config, DEVICE="cpu", LOSO_FOLDS=1),
              patch.multiple(ssvepformer_eKAN_config, BATCH_SIZE=4, EPOCHS=1)):
            train.train_experiment(data_file, run_dir, ("ssvepformer_eKAN",))
            evaluate.evaluate_experiment(data_file, run_dir)
        with np.load(run_dir / name / "artifacts" / "predictions.npz") as predictions:
            for count in (125, 250):
                np.testing.assert_array_equal(predictions[f"subject_ids_{count}"], [1])
                logits = predictions[f"logits_{count}"]
                assert logits.shape == (1, 4, 40) and np.isfinite(logits).all()

    summary = ("SSVEPformer eKAN CPU 冒烟通过。S1–S3 真实试次，两个窗长，输入 [9, 562]。\n"
               "B 样条基函数、反向梯度、共用训练、末轮权重加载及留出被试预测均通过。\n")
    (output / "ssvepformer_eKAN.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")


if __name__ == "__main__":
    main()
