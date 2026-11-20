# SSVEP 模型复现与比较

本项目使用同一份 Benchmark 裁窗数据和同一套 LOSO 协议，比较 SSVEPformer、SSVEPformer eKAN、FB-SSVEPformer、SpikeTFNet、EEG-Deformer、MTSNet 和 SsvepGPT。 [SSVEPformer 与 FB-SSVEPformer](https://arxiv.org/abs/2210.04172) · [EEG-Deformer](https://github.com/yi-ding-cs/EEG-Deformer) · [MTSNet 作者实现](https://github.com/lanzhen19/MTSNet/blob/main/MTSNet.py)

## 准备数据与运行

从项目根目录执行以下命令。服务器环境使用 Python 3.12 和 `requirements.txt` 中的 CUDA 12.8 依赖。

```text
python -m pip install -r requirements.txt
python -m data_preprocess.preprocess
python -m smoke_test.check_gpu
```

核对下文的 GPU 预热报告后，再开始正式训练与评估。

```text
python train.py
python evaluate.py
```

原始数据默认位于 `data/benchmark/`，需要 `S1.mat` 至 `S35.mat`、`64-channels.loc` 和 `Freq_Phase.mat`。在 `data_preprocess/config.py` 修改原始数据路径和 `WINDOW_SECONDS`。脚本生成 `data_preprocess/benchmark.npz`，其中含被试编号、每个试次的区块和目标索引、类别对应的频率与相位，以及各窗长的原始和滤波波形。

所有模型共用 250 Hz、九个通道和 `[0.64, 0.64+d)` 的截窗规则。各窗长独立截取，随后逐试次做 8–64 Hz、原型阶数为 4 的巴特沃斯双向滤波。FB 直接对同一原始裁窗做三个 Chebyshev-I 子带滤波。不同模型的输入变换在每个窗长下准备一次，供所有折复用。

| 模型 | 默认输入（省略批次轴） |
| --- | --- |
| `ssvepformer` | `[18, 281]`，复频谱实部与虚部沿通道轴拼接 |
| `fb-ssvepformer` | `[3, 18, 281]`，三个子带各自做复频谱变换 |
| `spiketfnet` | `[9, 采样点]` 和 `[9, 560]`，时域与复频谱双输入 |
| `eeg-deformer` | `[9, 采样点]`，默认 3 个 HCT 块 |
| `mtsnet` | `[9, 采样点]` 和 `[9, 560]`，分别编码时域与复频谱，再沿特征轴拼接并融合 |
| `ssvepgpt` | `[9, 采样点]` 和 `[9, 562]`，用联合自注意力编码时域短片段和各电极的整窗复频谱 |

SsvepGPT 借鉴 SleepGPT 的分域前馈和统一时频注意力，使用 Benchmark 标签和共用交叉熵从头训练。每 25 点生成一个时域 token，每个电极的整窗复频谱生成一个频域 token。频谱使用 8–64 Hz（含两端）、0.2 Hz 网格及 `1 / 窗内采样点数` 缩放。两个域共享电极嵌入，时域另加片段位置嵌入。默认宽度 96、4 个头、3 层，每层均允许跨域注意，前两层分域前馈，最后一层共享前馈。分域平均池化后拼接做 40 类分类。设置见 `configs/ssvepgpt_config.py`。[SleepGPT 官方实现](https://github.com/LordXX505/SleepGPT)

MTSNet 从同一滤波试次取得两路输入。频域分支将波形补零至 1250 点，在 8.0 至 63.8 Hz 每隔 0.2 Hz 取一个频点，乘以 `2 / 窗内采样点数` 后沿末轴拼接实部和虚部。该变换逐试次独立计算。模型默认使用两个时域块（核长 15、31 串联）、两个频域块（核长 31）和两个融合块（核长 31、15、7 并联）。频谱和模型参数见 `configs/mtsnet_config.py`。各折从头训练，使用共用交叉熵。

EEG-Deformer 的输入在 `configs/eeg-deformer_config.py` 中选择。`INPUT_TYPE = "time"` 保留时域基线。
设为 `"complex_channels"` 后，对每个已滤波的试次独立做实数 FFT，按原窗采样点数归一化，在 8–64 Hz 每隔 0.2 Hz 取一个频点（含两端）。九通道实部在前、九通道虚部在后，输入形状为 `[18, 281]`。设为 `"complex"` 时，每个通道沿最后一轴拼接实部和虚部，形状为 `[9, 562]`。切换表示时在共用配置中另设 `RESULT_NAME`，可以直接复用已有预处理文件。评估按该次实验保存的输入设置加载模型，ITR 仍按实际裁窗时长计算。

## 训练设置

`configs/config.py` 集中修改 `RESULT_NAME`、`MODELS`、LOSO 折数、随机种子，以及 batch、优化器、学习率和基础轮数。窗长在 `data_preprocess/config.py` 配置。首 `N` 折依次留出 S1 至 SN 作为最终测试，每折从其余被试中固定抽取约 20% 做验证。所有模型和窗长使用同一份划分，按验证 ACC 保存最佳权重。

FB 的子网预训练轮数和联合训练轮数在 `configs/fb-ssvepformer_config.py` 设置。各子网从本折训练被试学习，之后用各自的最佳验证权重开始联合训练。SpikeTFNet 的两个辅助损失权重在 `configs/spiketfnet_config.py` 设置。

`DEVICE = "cpu"` 使用 CPU。`DEVICE = "cuda"` 时，`CUDA_DEVICE_IDS = (0,)` 单卡顺序训练。设为 `(0, 1)` 且选择至少两个模型时，两个独立进程按 `MODELS` 顺序交替分配模型，每次每卡独立训练一个完整模型。只选一个模型并指定双卡时，使用 `DataParallel`。GPU 预热按同一分配规则逐模型测试真实 batch，结果写入 `smoke_test/output/gpu_report.json`。

### SsvepGPT 正式训练前的核对

在本地运行 `python -m smoke_test.check_ssvepgpt`，使用 S1–S3 的真实试次核对两个窗长的复频谱、跨域交互、反向传播和评估。结果写入 `smoke_test/output/ssvepgpt.txt`。

服务器准备好共用 NPZ 后，以正式实验的模型组合、设备编号和 batch 运行 `python -m smoke_test.check_gpu`。只测 SsvepGPT 单卡时，设置 `MODELS = ("ssvepgpt",)`、`CUDA_DEVICE_IDS = (0,)`。每个窗长先跑 3 个预热 batch、再计时 5 个，最后用 1 个 batch 记录算子，均取自首折训练被试，包含反向传播和优化器更新。

正式训练前核对 `smoke_test/output/gpu_report.json` 中的 `models.ssvepgpt`。默认单卡 batch 为 64，时域输入应为 `[64, 9, 125]` 或 `[64, 9, 250]`，频域输入为 `[64, 9, 562]`，输出为 FP32 的 `[64, 40]`。查看两个窗长的 `loss_and_gradients_finite`、batch 耗时中位数及峰值显存。核对 GPU 型号、确定性开关、cuBLAS workspace、FP32 精度设置，再结合 `observed_operators` 中的注意力算子及 `cuda_kernel_examples` 确认该设备实际使用的后端。CUDA 事件不可用时只能核对算子级记录。

### MTSNet 正式训练前的 GPU 核对

在服务器的 `configs/config.py` 中选定正式实验的模型、设备和 batch。仅测 MTSNet 单卡性能时，使用 `MODELS = ("mtsnet",)` 和 `CUDA_DEVICE_IDS = (0,)`。在目标 4090 D 上准备好共用 NPZ 后，先单独运行 `python -m smoke_test.check_gpu`。默认对每个窗长取首折训练被试的真实 batch，预热 3 次、计时 5 次，再单独采样一次算子。每次均执行前向、交叉熵、反向和优化器更新。

正式 LOSO 前，将 `smoke_test/output/gpu_report.json` 中的 `models.mtsnet` 发回核对。

- 默认 batch 为 64。单卡两路输入应为 `[64, 9, 125]` 或 `[64, 9, 250]`，以及 `[64, 9, 560]`。两个窗长的 logits 均应为该卡上的 `[64, 40]`，且 `loss_and_gradients_finite` 为真。
- 核对 `runtime` 中的 GPU 型号、设备编号、PyTorch、CUDA、cuDNN 版本。默认使用 FP32，矩阵乘法和 cuDNN 卷积精度均为 `ieee`，确定性算法和 cuDNN 确定性开启，cuDNN benchmark 关闭。
- 结合 `observed_operators` 和 `cuda_kernel_examples` 判断实际执行的卷积与矩阵算子。若 CUDA 事件不可用，报告只能提供算子级证据。按窗长比较训练 batch 耗时中位数与峰值显存，再决定是否开始正式训练。

## 结果与断点

`RESULT_NAME` 对应 `results/` 下的一个结果目录。实验根目录的 `report.md` 给出跨模型测试 ACC 与 ITR，每个模型目录保留训练汇总和详细产物。

```text
results/<RESULT_NAME>/
├── report.md
├── artifacts/run.json
└── <模型名>/
    ├── report.md
    ├── artifacts/
    │   ├── test_by_subject.csv
    │   └── predictions.npz
    └── 125pt_s01/
        ├── train.png
        └── artifacts/
            ├── training.log
            └── checkpoint.pt
```

每个 epoch 保存 Loss、ACC、学习率、训练耗时和训练加验证耗时，并更新曲线。同名实验重复执行 `python train.py` 会从上一个已提交的 epoch 恢复，已完成的折会跳过。FB 的 checkpoint 还记录训练阶段。数据、设备或训练配置有变时，使用新的 `RESULT_NAME`。

`evaluate.py` 只读取已完成的折和最佳验证权重。逐被试计算 ACC、ITR，再按模型和窗长汇总均值 ± 总体标准差。ITR 的选择时间取 `d + 0.5 s`，准确率低于或等于机会水平时记为 0。计划折数全完成后显示汇总值。

## 使用旧 EEG-Deformer 权重

需要复用旧运行时，在 `configs/config.py` 将 `LEGACY_EEG_RUN_DIR` 填为旧实验根目录，例如 `r"I:\jjh\SSVEP\EEG-Deformer\results\eeg_deformer_seed42"`。若旧预处理文件已移动，再填写 `LEGACY_EEG_DATA_FILE`。可以在 `MODELS` 中只训练其他模型，使用 `evaluate.py` 读取旧运行的 `artifacts/run.json`、原始预处理 NPZ 和各折 `checkpoint.pt`，并将结果写入 `eeg-deformer-legacy/`。评估会核对数据指纹、训练口径和各折被试。存在口径差异的旧结果在根报告中单列。

## 接入新模型

新增 `models/<模型名>.py` 和 `configs/<模型名>_config.py`，在后一个文件中提供 `SETTINGS` 字典，再将模型名写进 `MODELS`。新模型提供两个函数。

- `prepare_inputs(archive, count, settings)` 返回一个或多个 NumPy 输入数组组成的元组。每个数组的前两轴是 `[被试, 试次]`。从 `eeg_{count}` 读取共同滤波波形，从 `raw_{count}` 读取原始裁窗。需要 FFT 或子带滤波时可调用 `data_preprocess/features.py`。
- `build_model(shapes, num_classes, settings)` 接收各输入数组去掉前两轴后的形状，返回一个 `nn.Module`。`forward` 接收对应数量的输入张量，返回分类 logits 或包含 `logits` 的字典。联合损失可通过 `joint_loss(output, labels, settings)` 提供。FB 的阶段配置可参考其 `training_plan(settings)`。

本地可用 CPU 版 PyTorch 运行 `python -m smoke_test.check_pipeline`。它抽取 S1–S3 的少量真实试次，检查五个模型、两个窗长、断点恢复和旧权重的读取，并核对 MTSNet 的复频谱与梯度。冒烟产物放在 `smoke_test/output/`。
