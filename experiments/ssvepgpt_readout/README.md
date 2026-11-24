# 冻结 RMS SsvepGPT 的线性读出实验

这个实验检验末层频域 token 的平均操作是否限制了跨被试分类。实验入口独立于项目的模型列表。从项目根目录执行以下命令。

## 服务器运行

1. 准备原实验的 `results/ssvepgpt_rms/artifacts/run.json`、`results/ssvepgpt_rms/ssvepgpt/<窗长>pt_s<被试>/artifacts/checkpoint.pt` 和匹配的 `data_preprocess/benchmark.npz`。完整实验需要两个窗长、35 名被试的 70 份权重。Git 忽略数据和结果目录，服务器需要已有这些文件或单独传输。
2. 用真实 S01 折训练被试检查设备。默认按照项目的设备与精度设置，在 CUDA 模式下两种读出依次使用指定的单卡。

   ```bash
   python -m smoke_test.check_ssvepgpt_readout --gpu 0
   ```

   在 `smoke_test/output/ssvepgpt_readout/cuda_report.json` 核对设备、确定性、数值精度、主干注意力算子、分类头梯度和计时。两个窗长的公共特征均为 `[64, 37, 48]`，输出为 `[64, 40]`。本地 CPU 冒烟使用 `--cpu`，另含小样本端到端及续跑检查。

3. 运行两个窗长的全部折。中断后使用同一命令续跑。

   ```bash
   python -m experiments.ssvepgpt_readout --gpu 0
   ```

   默认写入 `results/ssvepgpt_readout`。如需先跑一折，追加 `--folds 1 --run-dir results/ssvepgpt_readout_pilot`。数据和源结果目录可用 `--data-file`、`--source-run` 指定。输出目录首次使用时应为空。

## 读出与评估

每折加载该窗长、该测试被试对应的 RMS 主干。主干使用评估模式，从最后一个 LayerNorm 后提取 token，按原有顺序保留。时域 token 取平均，每个试次的频域部分保留 9 个电极各 4 个频带的 36 个 token。特征逐试次计算，同一折在内存中提取一次，供两个分类头使用。

| 分类头 | 频域读出 | 线性层输入 | 可训练参数（含偏置） |
| --- | --- | ---: | ---: |
| A_mean | 36 个 token 取平均 | 96 | 3,880 |
| B_flat | 按电极、频带顺序展开 | 1,776 | 71,080 |

两个头都从随机初始化开始，用该折全部 34 名训练被试拟合。固定训练 100 轮，batch 64，seed 42，使用交叉熵、Adam、初始学习率 0.001、余弦退火和权重衰减 0.00001。每个头只在训练结束后评估留出被试。实验清单记录源配置、逐折权重校验和与运行环境。每折沿用项目原有的日志、曲线与断点格式。训练时间统计覆盖分类头更新，特征提取耗时未计入。

结果根目录的 `report.md` 汇总两个头的 ACC、ITR、同一被试的平均 ACC 差和胜平负折数。A 检验固定原表示后重训分类头的效果；B 允许每个电极、频带使用独立分类权重。解释 B 相对 A 的差异时，应同时考虑保留位置信息和分类头参数增加。该对照测量的是冻结主干末层特征的线性可读性；跨域注意力本身的贡献需要单独的对照实验。
