import numpy as np
import torch
from torch import nn


ARMS = ("A_mean", "B_flat")


def read_features(tokens, arm):
    # 第一个位置已是时域平均，其余位置按电极、频带排序。
    if arm == "A_mean":
        return torch.cat((tokens[:, 0], tokens[:, 1:].mean(1)), dim=1)
    if arm == "B_flat":
        return tokens.flatten(1)
    raise ValueError(f"未知读出方式 {arm}")


class Readout(nn.Module):
    def __init__(self, shape, num_classes, arm):
        super().__init__()
        positions, width = shape
        self.arm = arm
        self.linear = nn.Linear((2 if arm == "A_mean" else positions) * width, num_classes)

    def forward(self, tokens):
        return self.linear(read_features(tokens, self.arm))


def build_model(shapes, num_classes, settings):
    return Readout(shapes[0], num_classes, settings["arm"])


@torch.inference_mode()
def extract(backbone, inputs, batch_size, device):
    """沿原模型的 forward 提取末层特征，保持被试、试次顺序。"""
    eeg, spectrum = inputs
    subjects, trials, channels = eeg.shape[:3]
    freq_count = channels * backbone.spectrum_bands
    width = backbone.norm.normalized_shape[0]
    result = np.empty((subjects * trials, 1 + freq_count, width), dtype=np.float32)
    eeg = eeg.reshape(-1, *eeg.shape[2:])
    spectrum = spectrum.reshape(-1, *spectrum.shape[2:])

    def capture(module, args, tokens):
        time_mean = tokens[:, :-freq_count].mean(1, keepdim=True)
        features = torch.cat((time_mean, tokens[:, -freq_count:]), dim=1)
        result[start:start + len(tokens)] = features.cpu().numpy()

    backbone.eval().requires_grad_(False)
    with backbone.norm.register_forward_hook(capture):
        for start in range(0, len(eeg), batch_size):
            batch = slice(start, start + batch_size)
            backbone(torch.from_numpy(eeg[batch]).to(device), torch.from_numpy(spectrum[batch]).to(device))
    return result.reshape(subjects, trials, 1 + freq_count, width)
