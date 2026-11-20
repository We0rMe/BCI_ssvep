"""SSVEPformer，使用固定网格的 EfficientKAN 输出层。"""

import torch
from torch import nn
from torch.nn import functional as F

from models.ssvepformer import SSVEPformer, model_config, prepare_inputs


class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, cfg):
        super().__init__()
        size, order = cfg.grid_size, cfg.spline_order
        low, high = cfg.grid_range
        if size < 1 or order < 0 or low >= high:
            raise ValueError("样条网格需要正数个区间、非负次数和递增的范围")
        self.order = order
        step = (high - low) / size
        knots = low + step * torch.arange(-order, size + order + 1, dtype=torch.float32)
        self.register_buffer("knots", knots)

        self.base = nn.Linear(in_features, out_features, bias=False)
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, size + order))
        nn.init.normal_(self.base.weight, std=cfg.init_std)
        nn.init.normal_(self.spline_weight, std=cfg.init_std)

    def spline_basis(self, x):
        # 从各区间的指示函数递推 B 样条，每个输入特征只计算一次基函数。
        x = x.unsqueeze(-1)
        knots = self.knots
        basis = ((x >= knots[:-1]) & (x < knots[1:])).to(x.dtype)
        for k in range(1, self.order + 1):
            left = (x - knots[:-k-1]) / (knots[k:-1] - knots[:-k-1])
            right = (knots[k+1:] - x) / (knots[k+1:] - knots[1:-k])
            basis = left * basis[..., :-1] + right * basis[..., 1:]
        return basis

    def forward(self, x):
        basis = self.spline_basis(x)
        spline = F.linear(basis.flatten(1), self.spline_weight.flatten(1))
        return self.base(F.silu(x)) + spline


def build_model(shapes, num_classes, settings):
    cfg = model_config(shapes[0], num_classes, settings)
    model = SSVEPformer(cfg)
    output = model.classifier[-1]
    model.classifier[-1] = KANLinear(output.in_features, output.out_features, cfg)
    return model
