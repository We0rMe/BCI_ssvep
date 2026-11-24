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

原始数据默认位于 `data/benchmark/`，需要 `S1.mat` 至 `S35.mat`、`64-channels.loc` 和 `Freq_Phase.mat`。在 `data_preprocess/config.py` 修改原始数据路径和 `WINDOW_SECONDS`。运行 `python -m data_preprocess.preprocess` 可生成下表中的两个文件。

在 `configs/config.py` 中设置 `PREPROCESS_IDS`，训练和评估共用这项选择。默认 `(1,)` 使用原流程，`(2,)` 使用整试次滤波，`(1, 2)` 依次独立运行两次。表中的实验根目录由 `RESULT_NAME` 或 `--run-dir` 指定。

| 编号 | 预处理顺序 | 数据文件（位于 data_preprocess） | 训练、断点与评估目录 |
|---|---|---|---|
| 1 | 先截取分类窗，再在窗内滤波 | `benchmark.npz` | 实验根目录 |
| 2 | 先在整条 6 秒试次上滤波，再截取分类窗 | `benchmark_full_trial.npz` | 实验根目录下的 `preprocess_2/` |

两种数据使用相同的九个通道、250 Hz 采样率、试次标签和 `[0.64, 0.64+d)` 分类窗。常规输入采用原型阶数为 4 的 8–64 Hz 巴特沃斯双向滤波。编号 2 的滤波会用到分类窗前后的同一试次数据。FB 在准备模型输入时对原始波形应用自己的 Chebyshev-I 子带滤波器，并遵循所选的滤波与截窗顺序。

| 模型 | 默认输入（省略批次轴） |
| --- | --- |
| `ssvepformer` | `[9, 562]`，复频谱实部与虚部沿频率轴拼接 |
| `ssvepformer_eKAN` | `[9, 562]`，使用 eKAN 分类头 |
| `fb-ssvepformer` | `[3, 9, 562]`，三个子带各自做复频谱变换 |
| `spiketfnet` | `[9, 采样点]` 和 `[9, 560]`，时域与复频谱双输入 |
| `eeg-deformer` | `[9, 采样点]`，默认 3 个 HCT 块 |
| `mtsnet` | `[9, 采样点]` 和 `[9, 560]`，分别编码时域与复频谱，再沿特征轴拼接并融合 |
| `ssvepgpt` | `[9, 采样点]` 和 `[9, 562]`，用联合自注意力编码时域短片段和各电极的整窗复频谱 |
| `ssvepGPT_Linear` | `[9, 562]`，按电极顺序展开 RMS 归一化复频谱，用单个线性层输出 40 类 logits |

`ssvepGPT_Linear` 的模型和训练配方独立配置。每个试次用全部电极和采样点计算一个 RMS，再计算 8–64 Hz 整窗复频谱。Adam 的耦合 L2 系数为 `1e-5`，作用于权重和偏置。与现有 SsvepGPT RMS 结果对照时使用预处理 1、固定 100 轮和独立的结果目录。CPU 冒烟入口为 `python -m smoke_test.check_ssvepGPT_Linear`。

SsvepGPT 借鉴 SleepGPT 的分域前馈和统一时频注意力，使用 Benchmark 标签和共用交叉熵从头训练。每 25 点生成一个时域 token。每个电极在整窗上计算复频谱，保留 8–64 Hz（含两端）、0.2 Hz 网格及 `1 / 窗内采样点数` 缩放。默认将 281 个频点依次分成 71、70、70、70 点的四带，带内配对实部和虚部，各带共用线性投影并加上可学习的频带位置嵌入。九个电极共生成 36 个频域 token。两个域共享电极嵌入，时域另加片段位置嵌入。模型宽度 48、4 个头、3 层，每层均允许跨域注意，前两层分域前馈，最后一层共享前馈。分域平均池化后拼接做 40 类分类。模型配置中的 `spectrum_bands = 1` 可复用整谱投影，评估旧实验时按其运行清单构建模型。[SleepGPT 官方实现](https://github.com/LordXX505/SleepGPT)

MTSNet 从同一滤波试次取得两路输入。频域分支将波形补零至 1250 点，在 8.0 至 63.8 Hz 每隔 0.2 Hz 取一个频点，乘以 `2 / 窗内采样点数` 后沿末轴拼接实部和虚部。该变换逐试次独立计算。模型默认使用两个时域块（核长 15、31 串联）、两个频域块（核长 31）和两个融合块（核长 31、15、7 并联）。频谱和模型参数见 `configs/mtsnet_config.py`。各折从头训练，使用共用交叉熵。

EEG-Deformer 的输入在 `configs/eeg-deformer_config.py` 中选择。`INPUT_TYPE = "time"` 保留时域基线。
设为 `"complex_channels"` 后，对每个已滤波的试次独立做实数 FFT，按原窗采样点数归一化，在 8–64 Hz 每隔 0.2 Hz 取一个频点（含两端）。九通道实部在前、九通道虚部在后，输入形状为 `[18, 281]`。设为 `"complex"` 时，每个通道沿最后一轴拼接实部和虚部，形状为 `[9, 562]`。切换表示时用 `--run-dir` 指定新的结果目录，可以直接复用已有预处理文件。评估按该次实验保存的输入设置加载模型，ITR 仍按实际裁窗时长计算。

## 训练设置

每折留出一名被试用于测试，其余 34 名被试的全部试次参与训练。默认运行 35 折；将共用配置中的 `LOSO_FOLDS` 设为 N，只运行依次留出 S1 至 SN 的前 N 折，每折仍用其余 34 人训练。每折从头训练到配置的轮数，测试使用末轮权重。测试 ACC 和 ITR 由 `evaluate.py` 单独计算。

普通 SSVEPformer 使用固定学习率 0.001、SGD 动量 0.9、L2 系数 0.001、batch 128 和 100 轮训练。L2 惩罚由 SGD 的 `weight_decay` 实现。重训时请使用新的实验目录，以保留采用原训练配方的结果。

普通 SSVEPformer、FB 的各个子网及 eKAN 共用 `configs/ssvepformer_config.py` 中的主干配置。对默认复频谱输入，通道组合层和编码器在每个空间通道内，沿实部与虚部拼接而成的 562 维特征轴做 LayerNorm。分类头沿 240 个隐藏特征做 LayerNorm。卷积和线性层权重采用均值 0、标准差 0.01 的正态初始化，沿用编号 1 上验证过的基线。该标准差对应方差 0.0001，与作者公开稿写出的方差 0.01 有别。FB 的融合卷积和 eKAN 输出层的基础线性权重、样条系数也使用基线标准差。评估历史运行时按其清单选择归一化轴。[原文第 2.4、2.6 节](https://arxiv.org/pdf/2210.04172)

模型选择（`MODELS`）、设备、随机种子、LOSO 折数、确定性与 TF32 开关、ITR 附加时长和 GPU 预热与计时批数都在 `configs/config.py` 中设置。

每个模型的结构和训练超参数在 `configs/<模型名>_config.py`。可分别调整 batch、优化器（Adam 或 SGD）、动量、权重衰减、学习率、调度器（余弦退火或固定学习率）和训练轮数。FB 复用 SSVEPformer 的子网结构参数，在自己的配置文件中设置训练参数、各子网预训练轮数和联合训练轮数。联合训练从各子网末轮权重开始。SpikeTFNet 的辅助损失权重在其自身配置文件设置。

直接运行 `python train.py` 会训练 `MODELS` 中的模型。`--models` 可以为单次运行覆盖该列表。以下命令将两个模型的结果保存在同一实验目录，每个模型按自己的配置训练。评估读取该目录中实验清单所列的模型、训练配方、输入设置和被试划分。

```text
python train.py --models mtsnet eeg-deformer --run-dir results/comparison
python evaluate.py --run-dir results/comparison
```

配置 `DEVICE = "cpu"` 使用 CPU，`DEVICE = "cuda"` 时通过 `CUDA_DEVICE_IDS` 选择一张或两张显卡。例如 `CUDA_DEVICE_IDS = (0, 1)` 时，第 1、3 个模型在卡 0 上依次训练，第 2、4 个模型在卡 1 上依次训练，两张卡各用一个进程并行工作。只选一个模型时使用第一张卡。GPU 预热使用相同的模型选择与设备分配规则。

### SsvepGPT 正式训练前的核对

在本地运行 `python -m smoke_test.check_ssvepgpt`，使用 S1–S3 的真实试次核对两个窗长的复频谱、跨域交互、反向传播和评估。结果写入 `smoke_test/output/ssvepgpt.txt`。

服务器准备好共用 NPZ 后，以正式实验的模型组合、设备编号和 batch 运行 `python -m smoke_test.check_gpu`。只测 SsvepGPT 单卡时，运行 `python -m smoke_test.check_gpu --models ssvepgpt`，在共用配置中设置 `CUDA_DEVICE_IDS = (0,)`。每个窗长先跑 3 个预热 batch、再计时 5 个，最后用 1 个 batch 记录算子，均取自首折训练被试，包含反向传播和优化器更新。

正式训练前核对 `smoke_test/output/gpu_report.json` 中的 `models.ssvepgpt`。默认单卡 batch 为 64，时域输入应为 `[64, 9, 125]` 或 `[64, 9, 250]`，频域输入为 `[64, 9, 562]`，输出为 FP32 的 `[64, 40]`。查看两个窗长的 `loss_and_gradients_finite`、batch 耗时中位数及峰值显存。核对 GPU 型号、确定性开关、cuBLAS workspace、FP32 精度设置，再结合 `observed_operators` 中的注意力算子及 `cuda_kernel_examples` 确认该设备实际使用的后端。CUDA 事件不可用时只能核对算子级记录。

### MTSNet 正式训练前的 GPU 核对

在 `configs/config.py` 中选定 MTSNet 和设备，在 `configs/mtsnet_config.py` 中设置 batch。单卡使用 `CUDA_DEVICE_IDS = (0,)`。在目标 4090 D 上准备好共用 NPZ 后，先单独运行 `python -m smoke_test.check_gpu`。默认对每个窗长取首折训练被试的真实 batch，预热 3 次、计时 5 次，再单独采样一次算子。每次均执行前向、交叉熵、反向和优化器更新。

正式 LOSO 前，将 `smoke_test/output/gpu_report.json` 中的 `models.mtsnet` 发回核对。

- 默认 batch 为 64。单卡两路输入应为 `[64, 9, 125]` 或 `[64, 9, 250]`，以及 `[64, 9, 560]`。两个窗长的 logits 均应为该卡上的 `[64, 40]`，且 `loss_and_gradients_finite` 为真。
- 核对 `runtime` 中的 GPU 型号、设备编号、PyTorch、CUDA、cuDNN 版本。默认使用 FP32，矩阵乘法和 cuDNN 卷积精度均为 `ieee`，确定性算法和 cuDNN 确定性开启，cuDNN benchmark 关闭。
- 结合 `observed_operators` 和 `cuda_kernel_examples` 判断实际执行的卷积与矩阵算子。若 CUDA 事件不可用，报告只能提供算子级证据。按窗长比较训练 batch 耗时中位数与峰值显存，再决定是否开始正式训练。

## 结果与断点

在 `configs/config.py` 中设置 `RESULT_NAME`。训练默认写入 `results/<RESULT_NAME>/`，评估默认从同一目录读取。命令行 `--run-dir` 可临时覆盖该路径。实验根目录的 `report.md` 给出跨模型测试 ACC 与 ITR，每个模型目录保留训练汇总和详细产物。

```text
results/benchmark_loso34/
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

每个 epoch 保存 Loss、ACC、学习率和训练耗时，并更新曲线。同名实验重复执行 `python train.py` 会从上一个已提交的 epoch 恢复，已完成的折会跳过。FB 的 checkpoint 还记录训练阶段。数据、设备或训练配置有变时，更换 `RESULT_NAME` 或通过 `--run-dir` 选择新目录。

`evaluate.py` 只评估已完成的折。新实验加载末轮权重，历史实验按原协议加载最佳验证权重。逐被试计算 ACC、ITR，再按模型和窗长汇总均值 ± 总体标准差。ITR 的选择时间取 `d + 0.5 s`，准确率低于或等于机会水平时记为 0。计划折数全完成后显示汇总值。

## 使用旧 EEG-Deformer 权重

需要复用旧运行时，在 `configs/eeg-deformer_config.py` 将 `LEGACY_EEG_RUN_DIR` 填为旧实验根目录，例如 `r"I:\jjh\SSVEP\EEG-Deformer\results\eeg_deformer_seed42"`。若旧预处理文件已移动，再填写 `LEGACY_EEG_DATA_FILE`。可以通过 `--models`只训练其他模型，使用 `evaluate.py` 读取旧运行的 `artifacts/run.json`、原始预处理 NPZ 和各折 `checkpoint.pt`，并将结果写入 `eeg-deformer-legacy/`。评估会核对数据指纹、训练口径和各折被试。存在口径差异的旧结果在根报告中单列。

## 接入新模型

新增 `models/<模型名>.py` 和 `configs/<模型名>_config.py`，参照 MTSNet 配置提供本模型的训练参数和结构参数字典 `SETTINGS`，将模型名加入共用配置的 `MODELS` 或通过 `--models` 传入。新模型提供两个函数。

- `prepare_inputs(archive, count, settings)` 返回一个或多个 NumPy 输入数组组成的元组。每个数组的前两轴是 `[被试, 试次]`。从 `eeg_{count}` 读取共同滤波波形，从 `raw_{count}` 读取原始裁窗。需要 FFT 或子带滤波时可调用 `data_preprocess/features.py`。
- `build_model(shapes, num_classes, settings)` 接收各输入数组去掉前两轴后的形状，返回一个 `nn.Module`。`forward` 接收对应数量的输入张量，返回分类 logits 或包含 `logits` 的字典。联合损失可通过 `joint_loss(output, labels, settings)` 提供。FB 的阶段配置可参考其 `training_plan(settings)`。

本地可用 CPU 版 PyTorch 运行 `python -m smoke_test.check_pipeline`。它抽取 S1–S3 的少量真实试次，检查五个模型、两个窗长、断点恢复和旧权重的读取，并核对 MTSNet 的复频谱与梯度。冒烟产物放在 `smoke_test/output/`。
