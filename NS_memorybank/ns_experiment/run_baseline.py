"""小规模 NS FFNO baseline。

冒烟预算来自交接说明：10 个 epoch，每个 epoch 20 次更新。
它只检查协议、日志和 checkpoint，不代表方法已经有效。
"""

from __future__ import annotations

import csv
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

if os.environ.get("OMP_NUM_THREADS") == "0":
    os.environ["OMP_NUM_THREADS"] = "1"

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ns_experiment.common import (  # noqa: E402
    RunLog,
    parameter_count,
    relative_l2,
    rng_state,
    seed_everything,
    write_json,
    write_markdown,
)
from ns_experiment.data import (  # noqa: E402
    load_ns_bundle,
    population_mean_std,
    split_trajectory_indices,
    take_split,
)
from ns_experiment.model_memory import (  # noqa: E402
    autoregressive_fields,
    build_ffno,
    evaluate_autoregressive,
    forward_ffno,
    make_grid,
)
from ns_experiment.plots import (  # noqa: E402
    append_figure_notes,
    save_rollout_curves,
    save_training_curves,
    save_truth_prediction,
)

CFG = {
    "seed": 0,
    "epochs": 10,
    "updates_per_epoch": 20,
    "batch_size": 8,
    "learning_rate": 1e-3,
    "downsample": 8,
    "frames": 101,
    "rollout_steps": 100,
    "width": 32,
    "modes": 12,
    "layers": 4,
    "factor": 4,
    "grad_clip": 1.0,
    "device": "cuda",
    "data_path": str(ROOT / "data" / "ns_incom_inhom_2d_512-125.h5"),
}


def save_checkpoint(path: Path, model, optimizer, epoch: int, payload: dict) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": None if optimizer is None else optimizer.state_dict(),
            "epoch": epoch,
            "rng": rng_state(),
            **payload,
        },
        path,
    )


def prepare(device: torch.device):
    bundle = load_ns_bundle(Path(CFG["data_path"]), CFG["downsample"], CFG["frames"])
    if bundle["used_frames"] != CFG["frames"]:
        raise RuntimeError(
            f"需要 {CFG['frames']} 帧，文件只能提供 {bundle['used_frames']} 帧。"
        )
    if CFG["rollout_steps"] != CFG["frames"] - 1:
        raise RuntimeError("rollout_steps 必须等于 frames - 1。")
    split = split_trajectory_indices(bundle["states"].shape[0], CFG["seed"])
    parts = {name: take_split(bundle, split[name]) for name in ("train", "valid", "test")}
    state_mean, state_std = population_mean_std(parts["train"]["states"], (0, 1, 2, 3))
    force_mean, force_std = population_mean_std(parts["train"]["forces"], (0, 1, 2))
    height, width = parts["train"]["states"].shape[2:4]
    if CFG["modes"] > min(height, width) // 2 + 1:
        raise RuntimeError("modes 超过当前网格可表示的频率。")
    return bundle, split, parts, state_mean, state_std, force_mean, force_std, make_grid(height, width, device)


def save_baseline_figures(run_dir: Path, history: list[dict], test_steps: list[dict], truth, prediction) -> None:
    save_training_curves(history, run_dir / "training_curves.png")
    save_rollout_curves(test_steps, run_dir / "test_rollout_curves.png", "Test rollout")
    times = [float(row["time"]) for row in test_steps]
    indices = [index for index in (0, 9, 49, 99) if index < truth.shape[1]]
    speed_truth = np.linalg.norm(truth[0, :, :, :, :2].numpy(), axis=-1)
    speed_pred = np.linalg.norm(prediction[0, :, :, :, :2].numpy(), axis=-1)
    save_truth_prediction(
        run_dir / "test_speed.png",
        speed_truth,
        speed_pred,
        times,
        indices,
        "Test trajectory speed, 64x64 model grid",
    )
    save_truth_prediction(
        run_dir / "test_particles.png",
        truth[0, :, :, :, 2].numpy(),
        prediction[0, :, :, :, 2].numpy(),
        times,
        indices,
        "Test trajectory particles, 64x64 model grid",
    )
    append_figure_notes(
        run_dir / "summary.md",
        ["training_curves.png", "test_rollout_curves.png", "test_speed.png", "test_particles.png"],
    )


def plot_existing(run_dir: Path) -> None:
    import csv

    if not torch.cuda.is_available():
        raise RuntimeError("重画测试场需要 CUDA。")
    device = torch.device("cuda")
    with (run_dir / "history.csv").open(encoding="utf-8") as handle:
        history = list(csv.DictReader(handle))
    with (run_dir / "test_rollout.csv").open(encoding="utf-8") as handle:
        test_steps = list(csv.DictReader(handle))
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    config = checkpoint["config"]
    split = checkpoint["split"]
    bundle = load_ns_bundle(Path(config["data_path"]), config["downsample"], config["frames"])
    parts = {name: take_split(bundle, split[name]) for name in ("train", "valid", "test")}
    model = build_ffno(config).to(device)
    model.load_state_dict(checkpoint["model"])
    grid = make_grid(*parts["test"]["states"].shape[2:4], device)
    prediction, truth = autoregressive_fields(
        model,
        parts["test"]["states"],
        parts["test"]["forces"],
        checkpoint["state_mean"],
        checkpoint["state_std"],
        checkpoint["force_mean"],
        checkpoint["force_std"],
        grid,
        config["rollout_steps"],
        split["test"],
    )
    save_baseline_figures(run_dir, history, test_steps, truth, prediction)
    print("图像已写入：", run_dir, flush=True)


def train() -> Path:
    if CFG["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("要求 CUDA，但当前解释器看不到 GPU。")
    device = torch.device(CFG["device"])
    run_dir = ROOT / "results" / datetime.now().strftime("ns_baseline_%Y%m%d_%H%M%S")
    logger = RunLog(run_dir)
    try:
        print("结果目录：", run_dir, flush=True)
        seed_everything(CFG["seed"])
        bundle, split, parts, state_mean, state_std, force_mean, force_std, grid = prepare(device)
        write_json(run_dir / "config.json", CFG)
        write_json(
            run_dir / "data_identity.json",
            {
                "path": bundle["path"],
                "size_bytes": bundle["size_bytes"],
                "source_frames": bundle["source_frames"],
                "used_frames": bundle["used_frames"],
                "source_resolution": bundle["source_resolution"],
                "downsample": bundle["downsample"],
                "resolution": list(parts["train"]["states"].shape[2:4]),
                "channels": bundle["channels"],
                "channel_order_note": bundle["channel_order_note"],
                "config": {
                    key: value
                    for key, value in bundle["config"].items()
                    if key != "raw"
                },
                "latest_index_attr": bundle["latest_index_attr"],
                "time_start": float(bundle["times"][0, 0]),
                "time_at_rollout_end": float(bundle["times"][0, CFG["rollout_steps"]]),
                "time_file_end_used": float(bundle["times"][0, -1]),
                "saved_dt": float(bundle["times"][0, 1] - bundle["times"][0, 0]),
                "normalization": "train trajectories only, population mean/std, per channel",
                "nu_in_model_input": False,
                "nu_metadata": bundle["config"].get("NU"),
            },
        )
        write_json(run_dir / "split_ids.json", split)
        model = build_ffno(CFG).to(device)
        total, trainable = parameter_count(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=CFG["learning_rate"])
        print(f"参数量：{total}，可训练：{trainable}", flush=True)
        print("划分：", split, flush=True)
        state_mean_d = state_mean.to(device)
        state_std_d = state_std.to(device)
        force_mean_d = force_mean.to(device)
        force_std_d = force_std.to(device)
        train_states = parts["train"]["states"]
        train_forces = parts["train"]["forces"]
        history = []
        best_l2 = float("inf")
        best_epoch = 0
        fields = [
            "epoch",
            "train_normalized_mse",
            "train_relative_l2",
            "valid_relative_l2",
            "valid_velocity_relative_l2",
            "valid_mse",
            "seconds",
        ]
        with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for epoch in range(1, CFG["epochs"] + 1):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                clock = datetime.now()
                model.train()
                loss_sum = 0.0
                l2_sum = 0.0
                seen = 0
                for _update in range(CFG["updates_per_epoch"]):
                    batch = CFG["batch_size"]
                    trajectories = torch.randint(0, train_states.shape[0], (batch,))
                    times = torch.randint(0, train_states.shape[1] - 1, (batch,))
                    state = (train_states[trajectories, times].to(device) - state_mean_d) / state_std_d
                    target = (train_states[trajectories, times + 1].to(device) - state_mean_d) / state_std_d
                    force = (train_forces[trajectories].to(device) - force_mean_d) / force_std_d
                    optimizer.zero_grad(set_to_none=True)
                    prediction = forward_ffno(model, state, force, grid)
                    loss = (prediction - target).square().mean()
                    if not torch.isfinite(loss):
                        raise RuntimeError("训练损失出现 NaN/Inf。")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
                    optimizer.step()
                    loss_sum += float(loss.detach()) * batch
                    raw_prediction = prediction.detach() * state_std_d + state_mean_d
                    raw_target = target.detach() * state_std_d + state_mean_d
                    l2_sum += float(relative_l2(raw_prediction, raw_target).sum())
                    seen += batch
                valid = evaluate_autoregressive(
                    model,
                    parts["valid"]["states"],
                    parts["valid"]["forces"],
                    state_mean,
                    state_std,
                    force_mean,
                    force_std,
                    grid,
                    CFG["rollout_steps"],
                    split["valid"],
                )
                payload = {
                    "config": CFG,
                    "state_mean": state_mean,
                    "state_std": state_std,
                    "force_mean": force_mean,
                    "force_std": force_std,
                    "split": split,
                    "valid_relative_l2": valid["relative_l2"],
                }
                save_checkpoint(run_dir / "last.pt", model, optimizer, epoch, payload)
                if valid["relative_l2"] < best_l2:
                    best_l2 = valid["relative_l2"]
                    best_epoch = epoch
                    save_checkpoint(run_dir / "best.pt", model, optimizer, epoch, payload)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                seconds = (datetime.now() - clock).total_seconds()
                row = {
                    "epoch": epoch,
                    "train_normalized_mse": loss_sum / seen,
                    "train_relative_l2": l2_sum / seen,
                    "valid_relative_l2": valid["relative_l2"],
                    "valid_velocity_relative_l2": valid["velocity_relative_l2"],
                    "valid_mse": valid["mse"],
                    "seconds": seconds,
                }
                history.append(row)
                writer.writerow(row)
                handle.flush()
                print(
                    f"Epoch {epoch:02d}/{CFG['epochs']} | "
                    f"train MSE {row['train_normalized_mse']:.6f} | "
                    f"valid rollout L2 {row['valid_relative_l2']:.6f} | "
                    f"velocity L2 {row['valid_velocity_relative_l2']:.6f} | "
                    f"{seconds:.1f}s",
                    flush=True,
                )
        checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        test = evaluate_autoregressive(
            model,
            parts["test"]["states"],
            parts["test"]["forces"],
            state_mean,
            state_std,
            force_mean,
            force_std,
            grid,
            CFG["rollout_steps"],
            split["test"],
        )
        clock = parts["test"]["times"][0]
        for row in test["steps"]:
            row["time"] = float(clock[row["step"]])
        with (run_dir / "test_rollout.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(test["steps"][0].keys()))
            writer.writeheader()
            writer.writerows(test["steps"])
        summary = {
            "status": "completed",
            "best_epoch": best_epoch,
            "best_valid_relative_l2": best_l2,
            "test": {key: value for key, value in test.items() if key != "steps"},
            "parameters": {"total": total, "trainable": trainable},
            "history": history,
            "verified_scope": "一次 seed=0 的小预算 NS FFNO 冒烟训练",
            "not_verified": [
                "没有和记忆修正比较。",
                "没有第二种物理工况，没有做持续学习。",
                "10×20 次更新不能用来判断方法最终是否有效。",
                "联合 Relative L2 可能被 particles 的幅值主导。",
            ],
        }
        write_json(run_dir / "summary.json", summary)
        write_markdown(
            run_dir / "summary.md",
            "NS FFNO baseline 冒烟训练",
            [
                ("状态", [summary["status"], summary["verified_scope"]]),
                (
                    "数据",
                    [
                        f"路径：{bundle['path']}",
                        f"文件大小：{bundle['size_bytes']} 字节",
                        f"原始分辨率：{bundle['source_resolution']}，下采样倍数：{bundle['downsample']}",
                        f"使用帧数 / 文件帧数：{bundle['used_frames']} / {bundle['source_frames']}",
                        f"时间：{float(bundle['times'][0, 0])} 到 {float(bundle['times'][0, CFG['rollout_steps']])}，保存间隔 {float(bundle['times'][0, 1] - bundle['times'][0, 0])}",
                        f"划分：{split['rule']} train={split['train']} valid={split['valid']} test={split['test']}",
                        f"通道：{bundle['channels']}。{bundle['channel_order_note']}",
                        f"元数据 NU：{bundle['config'].get('NU')}。模型输入不含 NU，因为本文件只有一个黏性系数。",
                    ],
                ),
                (
                    "训练",
                    [
                        f"参数量：{total}",
                        "优化器：Adam，学习率 1e-3，损失为标准化状态的单步 MSE。",
                        f"预算：{CFG['epochs']} 个 epoch × {CFG['updates_per_epoch']} 次更新，batch {CFG['batch_size']}。",
                        "checkpoint 选择：验证集滚动联合 Relative L2 最小。",
                        f"最佳 epoch：{best_epoch}，验证滚动 L2：{best_l2:.8f}",
                        "每个完整 epoch 后写 last.pt；更优验证结果另写 best.pt。中断时没有当前 batch 的状态。",
                    ],
                ),
                (
                    "测试",
                    [
                        f"联合滚动 Relative L2：{test['relative_l2']:.8f}",
                        f"速度向量 Relative L2：{test['velocity_relative_l2']:.8f}",
                        f"vx / vy / particles：{test['relative_l2_velocity_x']:.8f} / {test['relative_l2_velocity_y']:.8f} / {test['relative_l2_particles']:.8f}",
                        f"最后一步联合 L2：{test['final_relative_l2']:.8f}",
                        f"原始尺度 MSE：{test['mse']:.8f}",
                    ],
                ),
                ("尚未验证", summary["not_verified"]),
            ],
        )
        prediction, truth = autoregressive_fields(
            model,
            parts["test"]["states"],
            parts["test"]["forces"],
            state_mean,
            state_std,
            force_mean,
            force_std,
            grid,
            CFG["rollout_steps"],
            split["test"],
        )
        save_baseline_figures(run_dir, history, test["steps"], truth, prediction)
        print("Summary：", run_dir / "summary.md", flush=True)
        return run_dir
    finally:
        logger.close()


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--plot":
        plot_existing(Path(sys.argv[2]))
    else:
        train()
