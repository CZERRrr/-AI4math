"""使用已有检查点诊断单步误差与滚动误差，不进行训练。"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ns_experiment.run_baseline_cohort import prepare
from ns_experiment.model_memory import (
    build_ffno,
    forward_ffno,
    make_grid,
)
from ns_experiment.common import summarize_fields, write_json


DEFAULT_RUN = (
    ROOT / "results"
    / "ns_cohort_B_20261018_111645_170902"
)

METHODS = {
    "ffno_teacher": "FFNO one-step (true input)",
    "persistence_teacher": "Persistence one-step (true input)",
    "ffno_rollout": "FFNO autoregressive",
    "persistence_initial": "Persistence from initial state",
}


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_finite(tensor, label):
    if not torch.isfinite(tensor).all().item():
        raise RuntimeError(f"{label} 出现 NaN/Inf。")


@torch.no_grad()
def diagnose(run_dir):
    run_dir = Path(run_dir).resolve()
    checkpoint_path = run_dir / "best.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if not torch.cuda.is_available():
        raise RuntimeError("没有检测到 CUDA，请在 AutoDL 远程终端运行。")

    device = torch.device("cuda")
    torch.set_num_threads(4)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    cfg = checkpoint["config"]

    output = run_dir / (
        "diagnostics_"
        + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)

    print("诊断目录：", output, flush=True)
    print("检查点 epoch：", checkpoint["epoch"], flush=True)
    print("GPU：", torch.cuda.get_device_name(0), flush=True)

    # 使用原训练程序的数据处理方式，核对数据与完整轨迹划分。
    parts, computed_norms, times, info = prepare(cfg)

    original_info = json.loads(
        (run_dir / "data_identity.json").read_text(encoding="utf-8")
    )
    for key in ("cache_sha256", "split_identities"):
        if info[key] != original_info[key]:
            raise RuntimeError(f"{key} 与原实验不一致，停止诊断。")

    if checkpoint["split"] != info["split_identities"]:
        raise RuntimeError("检查点中的轨迹划分与当前数据不一致。")

    norm_keys = (
        "state_mean", "state_std",
        "force_mean", "force_std",
    )
    norms = tuple(
        checkpoint[key].detach().cpu().float()
        for key in norm_keys
    )

    normalization_check = {}
    for key, saved, computed in zip(
        norm_keys, norms, computed_norms
    ):
        computed = computed.detach().cpu().float()
        difference = float((saved - computed).abs().max())
        normalization_check[key] = difference
        if not torch.allclose(
            saved, computed, rtol=1e-5, atol=1e-6
        ):
            raise RuntimeError(f"{key} 与训练时不一致。")

    mean, std, force_mean, force_std = (
        value.to(device) for value in norms
    )

    model = build_ffno(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    states = parts["test"]["states"]
    forces = parts["test"]["forces"]
    indices = parts["test"]["indices"]
    horizon = min(200, states.shape[1] - 1)

    grid = make_grid(states.shape[2], states.shape[3], device)
    per_trajectory = []

    # 分轨迹评估，避免少数轨迹被平均结果掩盖。
    for local_index, trajectory_index in enumerate(indices):
        raw = states[local_index:local_index + 1].to(device)
        force = forces[local_index:local_index + 1].to(device)
        normalized_force = (force - force_mean) / force_std

        initial = raw[:, 0]
        current = (initial - mean) / std

        for step in range(1, horizon + 1):
            truth = raw[:, step]
            previous_truth = raw[:, step - 1]

            # 每一步输入真实状态的 FFNO 单步预测。
            teacher_input = (previous_truth - mean) / std
            teacher_prediction = forward_ffno(
                model, teacher_input, normalized_force, grid
            ) * std + mean

            # 连续滚动：只在第一步使用真实状态。
            current = forward_ffno(
                model, current, normalized_force, grid
            )
            rollout_prediction = current * std + mean

            predictions = {
                "ffno_teacher": teacher_prediction,
                "persistence_teacher": previous_truth,
                "ffno_rollout": rollout_prediction,
                "persistence_initial": initial,
            }

            for method, prediction in predictions.items():
                check_finite(prediction, f"{method}, step={step}")
                metrics = summarize_fields(prediction, truth)
                if not all(np.isfinite(v) for v in metrics.values()):
                    raise RuntimeError("诊断指标出现 NaN/Inf。")
                per_trajectory.append({
                    "trajectory_index": int(trajectory_index),
                    "method": method,
                    "step": step,
                    "time": float(times[step]),
                    **metrics,
                })

            if step % 50 == 0:
                print(
                    f"测试轨迹 {trajectory_index}：{step}/{horizon} 步",
                    flush=True,
                )

    save_csv(output / "per_trajectory.csv", per_trajectory)

    metric_keys = [
        key for key in per_trajectory[0]
        if key not in (
            "trajectory_index", "method", "step", "time"
        )
    ]

    curves = {}
    aggregate_rows = []

    for method in METHODS:
        method_rows = [
            row for row in per_trajectory
            if row["method"] == method
        ]
        curves[method] = []

        for step in range(1, horizon + 1):
            rows = [row for row in method_rows if row["step"] == step]
            averaged = {
                key: float(np.mean([row[key] for row in rows]))
                for key in metric_keys
            }
            row = {
                "method": method,
                "step": step,
                "time": float(times[step]),
                **averaged,
            }
            curves[method].append(row)
            aggregate_rows.append(row)

    save_csv(output / "error_curves.csv", aggregate_rows)

    # 汇总前1、10、100、200步；另报告窗口最后一步。
    windows = [h for h in (1, 10, 100, 200) if h <= horizon]
    results = {}
    for method, rows in curves.items():
        results[method] = {}
        for horizon_window in windows:
            prefix = rows[:horizon_window]
            metrics = {
                key: float(np.mean([row[key] for row in prefix]))
                for key in metric_keys
            }
            metrics["final_relative_l2"] = (
                prefix[-1]["relative_l2"]
            )
            results[method][str(horizon_window)] = metrics

    # 第一步真实输入与滚动输入完全相同，应给出相同指标。
    first_step_gap = max(
        abs(
            curves["ffno_teacher"][0][key]
            - curves["ffno_rollout"][0][key]
        )
        for key in metric_keys
    )
    if first_step_gap > 1e-6:
        raise RuntimeError("FFNO 两种评估的第一步不一致。")

    report = {
        "status": "completed",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "test_trajectory_indices": indices,
        "horizon": horizon,
        "normalization_max_abs_differences": normalization_check,
        "first_step_metric_gap": first_step_gap,
        "results": results,
    }
    write_json(output / "summary.json", report)

    # 逐步误差曲线。
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    colors = {
        "ffno_teacher": "tab:blue",
        "persistence_teacher": "tab:green",
        "ffno_rollout": "tab:red",
        "persistence_initial": "tab:orange",
    }

    for ax, key, title in (
        (axes[0], "relative_l2", "Joint-state error"),
        (axes[1], "velocity_relative_l2", "Velocity error"),
    ):
        for method, label in METHODS.items():
            rows = curves[method]
            ax.plot(
                [row["time"] for row in rows],
                [row[key] for row in rows],
                label=label,
                color=colors[method],
            )
        ax.axvline(0.5, color="gray", linestyle="--", alpha=0.7)
        ax.set(
            title=title,
            xlabel="Physical time",
            ylabel="Per-step Relative L2",
        )
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(output / "diagnostic_errors.png", dpi=200)
    plt.close(fig)

    lines = [
        "# NS FFNO baseline 诊断",
        "",
        f"- 检查点：`{checkpoint_path}`",
        f"- 检查点 epoch：{checkpoint['epoch']}",
        f"- 测试轨迹：{indices}",
        "- 不训练，不修改原模型；数据划分与归一化已核对。",
        "- 前两种方法逐步使用真实状态，仅用于单步诊断。",
        "- 后两种方法只从初始状态开始预测。",
        "- 下表为前若干步的平均误差，不是最后一步误差。",
        "",
        "| 方法 | 步数 | 平均联合 L2 | 平均速度 L2 | 最后一步联合 L2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, label in METHODS.items():
        for window in windows:
            values = results[method][str(window)]
            lines.append(
                f"| {label} | {window} "
                f"| {values['relative_l2']:.6f} "
                f"| {values['velocity_relative_l2']:.6f} "
                f"| {values['final_relative_l2']:.6f} |"
            )

    lines.extend([
        "",
        "## 误差曲线",
        "",
        "曲线为各预测时刻的误差，纵轴采用对称对数刻度。"
        "虚线 t=0.5 标记训练时间窗口终点。",
        "",
        "![诊断误差曲线](diagnostic_errors.png)",
        "",
        "## 如何判断",
        "",
        "- 单步 FFNO 应与单步保持不变比较。",
        "- 连续滚动 FFNO 应与全程保持初始状态比较。",
        "- 单步较准而滚动迅速变差，说明滚动稳定性需要改进。",
        "- 单步也明显较差，需要继续检查泛化、训练与输入设计。",
        "- 单步误差是在真实输入分布上计算，不能直接等同于"
        "模型滚动时面对的输入分布。",
        "- 分通道及逐轨迹指标见 CSV 与 summary.json。",
    ])
    (output / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print("\n诊断完成。", flush=True)
    for method, label in METHODS.items():
        values = results[method]["100"]
        print(
            f"{label} | 前100步平均联合L2="
            f"{values['relative_l2']:.6f} | "
            f"速度L2={values['velocity_relative_l2']:.6f}",
            flush=True,
        )
    print("Summary：", output / "summary.md", flush=True)


if __name__ == "__main__":
    diagnose(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RUN)