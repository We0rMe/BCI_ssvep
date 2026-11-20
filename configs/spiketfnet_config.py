SETTINGS = {
    "lambda_spectral": 0.3,  # 频域分支辅助交叉熵权重
    "lambda_temporal": 0.3,  # 时域分支辅助交叉熵权重
    "input_type": "complex",  # 保留原时域输入，频域沿特征轴拼接实部和虚部
    "fft_resolution": 0.2,
    "spectrum_hz": (8.0, 64.0),
    "include_upper_bin": False,
    "fft_scale": 2.0,  # FFT × 2 / 窗内采样点数
    "dropout": 0.5,
    "temporal_backbone_dropout": 0.1,
    "spectral_depth": 2,
    "spectral_kernel": 31,
    "spectral_head_multiplier": 6,
    "spectral_init_std": 0.01,
    "temporal_width": 64,
    "stem_kernel": 31,
    "residual_kernel": 7,
    "temporal_readout_bins": 32,
    "dilations": (1, 2, 4),
    "attention_heads": 4,
    "attention_scale": 0.125,
    "embedding_dim": 128,
    "fusion_dim": 128,
    "gate_hidden_dim": 128,
    "simulation_steps": 4,
    "lif_tau": 2.0,
    "lif_threshold": 1.0,
    "lif_reset": 0.0,
    "lif_decay_input": False,
    "surrogate_alpha": 4.0,
    "detach_reset": False,
    "t2_conv_execution": "split_bias",  # 膨胀卷积采用已验证的确定性路径
}
