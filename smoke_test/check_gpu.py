import json
import statistics
from time import perf_counter

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, supported_activities

from configs import config
from data_preprocess import config as data_config
from data_preprocess.preprocess import window_samples
from train import (
    batch_step, check_archive, configure_runtime, input_shapes, loso_splits, make_optimizer, experiment_args, model_devices,
    model_module, parallel_model, prepare_inputs, runtime_details, model_config, set_seed, subject_loader, training_settings,
)


def probe(module, options, inputs, labels, subject_ids, split, device, cuda_ids, settings):
    set_seed(settings["SEED"], cuda_ids)
    shuffle = torch.Generator().manual_seed(settings["SEED"])
    batch_size = settings["BATCH_SIZE"]
    loader = subject_loader(inputs, labels, subject_ids, split["train"], batch_size, device, shuffle)
    if len(loader.dataset) // batch_size < config.GPU_WARMUP_BATCHES + config.GPU_PROBE_BATCHES + 1:
        raise ValueError("训练样本不足以提供完整的预热和计时 batch")
    batches = iter(loader)
    shapes = input_shapes(inputs)
    model = module.build_model(shapes, data_config.NUM_CLASSES, options).to(device)
    network = parallel_model(module, model, cuda_ids)
    optimizer = make_optimizer(model, settings)
    auxiliary_loss = None
    if hasattr(module, "joint_loss"):
        auxiliary_loss = lambda output, targets: module.joint_loss(output, targets, options)

    def synchronize():
        for index in cuda_ids:
            torch.cuda.synchronize(index)

    for _ in range(config.GPU_WARMUP_BATCHES):
        batch_step(network, next(batches), device, optimizer, auxiliary_loss)
    synchronize()
    for index in cuda_ids:
        torch.cuda.reset_peak_memory_stats(index)

    durations = []
    for _ in range(config.GPU_PROBE_BATCHES):
        batch = next(batches)
        started = perf_counter()
        loss, _, _ = batch_step(network, batch, device, optimizer, auxiliary_loss)
        synchronize()
        durations.append(1000 * (perf_counter() - started))
        if not torch.isfinite(loss):
            raise FloatingPointError("计时 batch 出现非有限损失")
    memory = {f"cuda:{index}": round(torch.cuda.max_memory_allocated(index) / 1024 ** 2) for index in cuda_ids}

    observed = {}

    def record_output(_module, _inputs, output):
        logits = output["logits"] if isinstance(output, dict) else output
        observed[str(logits.device)] = {"shape": list(logits.shape), "dtype": str(logits.dtype)}

    hook = model.register_forward_hook(record_output)
    activities = [ProfilerActivity.CPU]
    if cuda_ids and ProfilerActivity.CUDA in supported_activities():
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities, acc_events=True) as trace:
        loss, _, _ = batch_step(network, next(batches), device, optimizer, auxiliary_loss)
        synchronize()
    hook.remove()
    if not torch.isfinite(loss) or any(
        parameter.grad is None or not torch.isfinite(parameter.grad).all() for parameter in model.parameters()
    ):
        raise FloatingPointError("预热检查出现无效损失或参数梯度")

    operators = sorted({event.key for event in trace.key_averages()
                        if any(name in event.key.lower() for name in
                               ("convolution", "cudnn", "aten::bmm", "aten::mm", "scaled_dot_product"))
                        and event.key.startswith("aten::")})
    events = [event for event in trace.events() if event.device_type == torch.autograd.DeviceType.CUDA
              and not event.is_user_annotation]
    events.sort(key=lambda event: event.device_time_total, reverse=True)
    return {
        "input_shapes": [[batch_size, *shape] for shape in shapes], "logits_per_device": observed,
        "loss_and_gradients_finite": True, "median_train_batch_ms": round(statistics.median(durations), 2),
        "peak_allocated_memory_mib": memory, "observed_operators": operators,
        "cuda_activity_enabled": ProfilerActivity.CUDA in activities,
        "cuda_kernel_examples": list(dict.fromkeys(event.name[:180] for event in events))[:5],
    }


def main():
    names, _ = experiment_args()
    if config.DEVICE != "cuda" or config.GPU_WARMUP_BATCHES < 0 or config.GPU_PROBE_BATCHES < 1:
        raise ValueError("GPU 预热须设置 CUDA，且至少有一个计时 batch")
    _, cuda_ids = configure_runtime()
    assignments = model_devices(names, cuda_ids)
    counts = window_samples(data_config.WINDOW_SECONDS)
    report = {"models": {}}
    with np.load(data_config.OUTPUT_FILE, allow_pickle=False) as archive:
        check_archive(archive, counts)
        subject_ids, labels = archive["subject_ids"], archive["labels"]
        split = loso_splits(subject_ids, 1)[0]
        for name in names:
            model_cfg = model_config(name)
            settings = training_settings(model_cfg)
            module = model_module(name)
            options = model_cfg.SETTINGS
            device, assigned = configure_runtime(assignments[name])
            results = {"warmup_batches": config.GPU_WARMUP_BATCHES, "timed_batches": config.GPU_PROBE_BATCHES,
                       "runtime": runtime_details(device, assigned), "windows": {}}
            for count in counts:
                print(f"正在检查 {name} / {count}pt / {assigned}", flush=True)
                inputs = prepare_inputs(module, archive, count, options)
                results["windows"][str(count)] = probe(module, options, inputs, labels, subject_ids,
                                                       split, device, assigned, settings)
                del inputs
            report["models"][name] = results
    output = data_config.PROJECT_DIR / "smoke_test" / "output" / "gpu_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"GPU 检查已保存到 {output}")


if __name__ == "__main__":
    main()
