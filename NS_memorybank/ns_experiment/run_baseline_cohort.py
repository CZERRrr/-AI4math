"""多缓存 NS baseline：训练 t<=0.5，测试最多200步。"""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

if os.environ.get("OMP_NUM_THREADS") == "0":
    os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ns_experiment.common import seed_everything, rng_state, write_json
from ns_experiment.data import load_cached_trajectories
from ns_experiment.model_memory import (
    build_ffno,
    forward_ffno,
    make_grid,
    evaluate_autoregressive,
)

CACHE = ROOT / "data/cache/nu0.01_512_down8"

# 固定三个已完成的缓存，不自动加入仍在下载的文件。
CACHES = [
    CACHE / f"ns_incom_inhom_2d_512-{s}_down8_f201.pt"
    for s in (125, 241, 194)
]


class Tee:
    def __init__(self, original, log):
        self.original = original
        self.log = log

    def write(self, text):
        self.original.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)

    def flush(self):
        self.original.flush()
        self.log.flush()


def atomic_save(path, payload):
    temp = path.with_suffix(".partial.pt")
    torch.save(payload, temp)
    temp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare(cfg):
    files = [Path(p).resolve() for p in cfg["cache_files"]]
    metadata = []
    hashes = []

    for path in files:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
        config = payload["config"]

        if not math.isclose(float(config["NU"]), 0.01):
            raise ValueError(f"工况 NU 不匹配：{path}")

        metadata.append(config)
        hashes.append(sha256(path))

    condition_fields = (
        "NU",
        "velocity_extrapolation",
        "particle_extrapolation",
        "force_extrapolation",
        "force_scale",
        "force_smoothness",
        "smoothness",
        "scale",
        "domain_size",
        "grid_size",
        "DT",
        "frame_int",
    )

    checked = []
    missing = []

    for key in condition_fields:
        if all(key in config for config in metadata):
            values = {
                json.dumps(config[key], sort_keys=True)
                for config in metadata
            }
            if len(values) != 1:
                raise ValueError(f"不同分片的工况不一致：{key}")
            checked.append(key)
        else:
            missing.append(key)

    records = load_cached_trajectories(files)

    split_sizes = {
        12: (8, 2, 2),
        16: (10, 3, 3),
        20: (12, 4, 4),
    }

    if len(records) not in split_sizes:
        raise ValueError(
            f"当前脚本支持12/16/20条，实际为{len(records)}条"
        )

    keys = [
        record["identity"]["trajectory_key"]
        for record in records
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("出现重复轨迹")

    states = torch.stack([
        record["states"][:201].float()
        for record in records
    ])
    forces = torch.stack([
        record["forces"].float()
        for record in records
    ])
    times = torch.stack([
        torch.as_tensor(record["times"][:201]).double()
        for record in records
    ])

    n, frames, height, width, channels = states.shape

    if (
        (frames, height, width, channels) != (201, 64, 64, 3)
        or forces.shape != (n, height, width, 2)
    ):
        raise ValueError(
            f"缓存尺寸异常：states={states.shape}，"
            f"forces={forces.shape}"
        )

    if not all(
        torch.isfinite(array).all()
        for array in (states, forces, times)
    ):
        raise ValueError("缓存中存在 NaN/Inf")

    if not torch.allclose(times, times[:1]):
        raise ValueError("不同轨迹的时间轴不一致")

    expected_dt = torch.full_like(times.diff(), 0.005)
    if not torch.allclose(
        times.diff(),
        expected_dt,
        atol=1e-7,
        rtol=1e-4,
    ):
        raise ValueError("保存时间间隔不是0.005")

    if abs(times[0, 0].item()) > 1e-7:
        raise ValueError("预期轨迹从 t=0 开始")

    order = np.random.default_rng(
        cfg["seed"]
    ).permutation(n).tolist()

    ntrain, nvalid, ntest = split_sizes[n]

    split = {
        "train": order[:ntrain],
        "valid": order[ntrain:ntrain + nvalid],
        "test": order[ntrain + nvalid:],
    }

    identities = [record["identity"] for record in records]

    parts = {}
    for name, indices in split.items():
        parts[name] = {
            "states": states[indices],
            "forces": forces[indices],
            "indices": indices,
        }

    # 只用训练轨迹的前101帧拟合状态统计量。
    train = parts["train"]["states"][:, :cfg["train_frames"]]
    train_force = parts["train"]["forces"]

    norms = (
        train.mean((0, 1, 2, 3)),
        train.std(
            (0, 1, 2, 3), unbiased=False
        ).clamp_min(1e-6),
        train_force.mean((0, 1, 2)),
        train_force.std(
            (0, 1, 2), unbiased=False
        ).clamp_min(1e-6),
    )

    info = {
        "cache_files": [str(path) for path in files],
        "cache_sha256": hashes,
        "identities": identities,
        "split": split,
        "split_identities": {
            name: [identities[i] for i in indices]
            for name, indices in split.items()
        },
        "shape": list(states.shape),
        "times": times[0].tolist(),
        "source_configs": metadata,
        "condition_fields_checked": checked,
        "condition_fields_absent_or_incomplete": missing,
    }

    return parts, norms, times[0], info


def window_metrics(curves, steps):
    section = curves[:steps]

    result = {
        key: sum(float(row[key]) for row in section) / steps
        for key in section[0]
        if key not in ("step", "time")
    }
    result["final_relative_l2"] = float(
        section[-1]["relative_l2"]
    )

    return result


def evaluate(model, part, norms, grid, steps, times):
    result = evaluate_autoregressive(
        model,
        part["states"],
        part["forces"],
        *norms,
        grid,
        steps,
        part["indices"],
    )

    curves = result["steps"]

    for row in curves:
        if not all(
            math.isfinite(float(value))
            for value in row.values()
        ):
            raise RuntimeError(
                "滚动评估出现NaN/Inf，停止并检查预测稳定性"
            )
        row["time"] = float(times[int(row["step"])])

    return curves


def save_summary(path, report):
    write_json(path / "summary.json", report)

    lines = [
        "# NS FFNO 多轨迹实验",
        "",
        f"- 状态：{report['status']}",
        f"- 预算：{report['config']['budget']}。",
        "- epoch表示固定20次更新，不是完整数据遍历。",
        "- 输入：[vx,vy,particles,fx,fy,x,y]。",
        "- 输出：[vx,vy,particles]。",
        "- 训练与状态归一化：仅训练轨迹前101帧，t=0～0.5。",
        "- 静态外力归一化：仅训练轨迹。",
        "- checkpoint选择：验证集前100步联合Relative L2最小。",
        "- 测试200步到t=1；t>0.5超出训练时间窗口。",
        "- 未加入记忆修正或CL；仅一个seed。",
        "",
        "## 配置",
        "",
        "```json",
        json.dumps(report["config"], indent=2),
        "```",
        "",
        f"- 参数量：{report.get('parameters')}",
        f"- 划分数量：{report.get('split_sizes')}",
        f"- 最近完成epoch：{report.get('completed_epoch', 0)}",
        f"- 最佳epoch：{report.get('best_epoch', 0)}",
        "",
        "## 测试",
        "",
        "| 步数 | 平均联合L2 | 平均速度L2 | 最后一步联合L2 | MSE |",
        "|---:|---:|---:|---:|---:|",
    ]

    for horizon, metrics in report.get("test", {}).items():
        lines.append(
            f"| {horizon} "
            f"| {metrics['relative_l2']:.6f} "
            f"| {metrics['velocity_relative_l2']:.6f} "
            f"| {metrics['final_relative_l2']:.6f} "
            f"| {metrics['mse']:.6g} |"
        )

    lines.extend([
        "",
        "- 完整分通道结果见summary.json。",
        "- 逐步结果见test_rollout.csv。",
        "- 来源、工况检查与划分见data_identity.json、split_ids.json。",
        f"- 本次调用耗时：{report.get('seconds', 0):.2f}秒。",
    ])

    if "error" in report:
        lines.extend(["", f"- 错误：{report['error']}"])

    (path / "summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

def save_experiment_figures(run_dir):
    """读取本次已保存的 CSV，画图并嵌入本次 summary.md。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(exist_ok=True)

    def read_table(filename):
        with (run_dir / filename).open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"{filename} 没有数据。")
        return {
            key: np.asarray(
                [float(row[key]) for row in rows],
                dtype=np.float64,
            )
            for key in rows[0]
        }

    def decorate(ax, title, xlabel, ylabel):
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)

    images = []

    def save(fig, filename, title, description):
        fig.tight_layout()
        fig.savefig(
            figure_dir / filename,
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)
        images.append((filename, title, description))
        print("图片已保存：", figure_dir / filename, flush=True)

    with plt.rc_context({"font.size": 10}):
        # 1. 训练损失与验证误差
        history = read_table("history.csv")
        epochs = history["epoch"]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        axes[0].plot(epochs, history["train_mse"])
        axes[0].set_yscale("symlog", linthresh=1e-8)
        decorate(
            axes[0],
            "One-step training loss",
            "Epoch",
            "Normalized MSE",
        )

        for horizon in (1, 10, 100):
            axes[1].plot(
                epochs,
                history[f"valid_{horizon}_l2"],
                label=f"First {horizon} step(s): mean L2",
            )

        best_index = int(np.argmin(history["valid_100_l2"]))
        best_epoch = int(epochs[best_index])
        axes[1].axvline(
            best_epoch,
            color="black",
            linestyle="--",
            alpha=0.6,
            label=f"Best epoch: {best_epoch}",
        )
        axes[1].set_yscale("symlog", linthresh=1e-4)
        decorate(
            axes[1],
            "Validation autoregressive error",
            "Epoch",
            "Relative L2 (symlog scale)",
        )
        axes[1].legend(fontsize=8)

        save(
            fig,
            "training_validation.png",
            "训练与验证曲线",
            "左图为标准化状态的单步训练 MSE；右图为验证集前 "
            "1、10、100 步的平均 Relative L2。虚线标记按验证 "
            "100 步误差选择的最佳 epoch。纵轴采用对称对数刻度。",
        )

        # 2. 测试集中，每个预测时刻各自的误差
        test = read_table("test_rollout.csv")
        times = test["time"]

        def mark_boundary(ax):
            if times.min() < 0.5 < times.max():
                ax.axvline(
                    0.5,
                    color="gray",
                    linestyle="--",
                    label="Training time window ends",
                )

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        for key, label in (
            ("relative_l2", "Joint state"),
            ("velocity_relative_l2", "Velocity vector"),
        ):
            axes[0].plot(times, test[key], label=label)

        mark_boundary(axes[0])
        decorate(
            axes[0],
            "Test error at each forecast step",
            "Physical time",
            "Relative L2",
        )
        axes[0].legend(fontsize=8)

        axes[1].plot(
            times, test["mse"],
            color="tab:red", label="Raw-state MSE",
        )
        mark_boundary(axes[1])
        decorate(
            axes[1],
            "Test MSE at each forecast step",
            "Physical time",
            "MSE in raw units",
        )
        axes[1].legend(fontsize=8)

        save(
            fig,
            "test_step_errors.png",
            "测试逐步误差",
            "每个点表示该预测时刻在测试轨迹上的平均误差，"
            "不是从起点累计的平均误差。t=0.5 的虚线表示训练"
            "时间窗口终点，之后的预测超出训练使用的时间范围。",
        )

        # 3. 分通道误差与累计平均误差
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        for channel, label in (
            ("velocity_x", "Velocity x"),
            ("velocity_y", "Velocity y"),
            ("particles", "Particles"),
        ):
            key = f"relative_l2_{channel}"
            if key in test:
                axes[0].plot(times, test[key], label=label)

        mark_boundary(axes[0])
        decorate(
            axes[0],
            "Per-channel test error",
            "Physical time",
            "Relative L2",
        )
        axes[0].legend(fontsize=8)

        for key, label in (
            ("relative_l2", "Joint state"),
            ("velocity_relative_l2", "Velocity vector"),
        ):
            values = test[key]
            cumulative_mean = np.cumsum(values) / np.arange(
                1, len(values) + 1
            )
            axes[1].plot(
                times, cumulative_mean, label=label
            )

        mark_boundary(axes[1])
        decorate(
            axes[1],
            "Mean error over steps 1 to current step",
            "Physical time",
            "Cumulative mean Relative L2",
        )
        axes[1].legend(fontsize=8)

        save(
            fig,
            "test_channels_and_mean.png",
            "分通道与累计平均误差",
            "左图分别展示两个速度分量和 particles 的误差。"
            "右图为从第 1 步到当前步的平均误差，其第 "
            "10、100、200 步对应 summary 表格中的平均滚动指标。",
        )

    # 将图片引用写入本次实验的 Markdown。
    # 重复画图时替换旧图片区，不重复追加。
    summary_path = run_dir / "summary.md"
    start_marker = "<!-- AUTO_FIGURES_START -->"
    end_marker = "<!-- AUTO_FIGURES_END -->"

    text = summary_path.read_text(encoding="utf-8")
    if start_marker in text:
        before, remainder = text.split(start_marker, 1)
        _, separator, after = remainder.partition(end_marker)
        text = before + (after if separator else "")

    lines = [start_marker, "", "## 实验结果图", ""]
    for filename, title, description in images:
        lines.extend([
            f"### {title}",
            "",
            description,
            "",
            f"![{title}](figures/{filename})",
            "",
        ])
    lines.append(end_marker)

    summary_path.write_text(
        text.rstrip() + "\n\n" + "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print("图片已加入：", summary_path, flush=True)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--budget", choices=["A", "B"], default="A"
    )
    parser.add_argument(
        "--prediction-mode",
        choices=["direct", "increment"],
        default="direct",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    cfg = {
        "budget": args.budget,
        "prediction_mode": args.prediction_mode,
        "seed": 0,
        "epochs": 10 if args.budget == "A" else 100,
        "updates_per_epoch": 20,
        "batch_size": 8,
        "learning_rate": 1e-3,
        "train_frames": 101,
        "validation_steps": [1, 10, 100],
        "test_steps": [1, 10, 100, 200],
        "width": 32,
        "modes": 12,
        "layers": 4,
        "factor": 4,
        "grad_clip": 1.0,
        "device": "cuda",
        "cache_files": [str(path) for path in CACHES],
    }

    if args.resume:
        run_dir = args.resume.resolve()
        cfg = json.loads(
            (run_dir / "config.json").read_text()
        )
    else:
        name = datetime.now().strftime(
            f"ns_cohort_{args.prediction_mode}_{args.budget}_%Y%m%d_%H%M%S_%f"
        )
        run_dir = ROOT / "results" / name
        run_dir.mkdir(parents=True, exist_ok=False)
        write_json(run_dir / "config.json", cfg)

    handle = (run_dir / "run.log").open(
        "a", encoding="utf-8", buffering=1
    )
    stdout, stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(stdout, handle)
    sys.stderr = Tee(stderr, handle)

    start_clock = time.perf_counter()
    report = {"status": "running", "config": cfg}

    try:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA不可用，停止；不自动改CPU"
            )

        device = torch.device("cuda")
        torch.set_num_threads(4)
        seed_everything(cfg["seed"])

        print("结果目录：", run_dir, flush=True)
        print(
            "预测模式：", cfg.get("prediction_mode", "direct"),
            flush=True,
        )
        print(
            "GPU：", torch.cuda.get_device_name(0),
            flush=True,
        )

        parts, norms, times, info = prepare(cfg)

        if args.resume:
            old = json.loads(
                (run_dir / "data_identity.json").read_text()
            )
            current = json.loads(json.dumps(
                info,
                default=lambda value: (
                    value.item()
                    if isinstance(value, np.generic)
                    else str(value)
                ),
            ))
            if old != current:
                raise RuntimeError(
                    "数据或划分变化，不能恢复到同一个实验"
                )
        else:
            write_json(
                run_dir / "data_identity.json", info
            )
            write_json(
                run_dir / "split_ids.json",
                info["split_identities"],
            )

        sizes = {
            name: len(part["states"])
            for name, part in parts.items()
        }

        print("轨迹划分：", sizes, flush=True)
        print(
            "工况已比较字段：",
            info["condition_fields_checked"],
            flush=True,
        )
        print(
            "元数据未覆盖字段：",
            info["condition_fields_absent_or_incomplete"],
            flush=True,
        )

        model = build_ffno(cfg).to(device)
        grid = make_grid(64, 64, device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg["learning_rate"],
        )

        report.update(
            parameters=sum(
                parameter.numel()
                for parameter in model.parameters()
            ),
            split_sizes=sizes,
        )
        report["environment"] = {
            "torch": str(torch.__version__),
            "gpu": torch.cuda.get_device_name(0),
        }

        sm, ss, fm, fs = [
            value.to(device) for value in norms
        ]

        # 真实训练样本前后向检查，不执行参数更新。
        x = (
            parts["train"]["states"][:1, 0].to(device) - sm
        ) / ss
        force = (
            parts["train"]["forces"][:1].to(device) - fm
        ) / fs
        y = (
            parts["train"]["states"][:1, 1].to(device) - sm
        ) / ss

        probe = forward_ffno(model, x, force, grid)

        if (
            probe.shape != y.shape
            or not torch.isfinite(probe).all()
        ):
            raise RuntimeError("前向检查失败")

        (probe - y).square().mean().backward()

        if not any(
            parameter.grad is not None
            and parameter.grad.abs().sum() > 0
            for parameter in model.parameters()
        ):
            raise RuntimeError("反向检查失败")

        optimizer.zero_grad(set_to_none=True)

        print(
            "CUDA前后向通过；训练t=0～0.5，测试到t=1.0。",
            flush=True,
        )

        if args.check_only:
            report["status"] = "check_passed"
            save_summary(run_dir, report)
            return

        history = []
        best = float("inf")
        best_epoch = 0
        completed = 0

        if args.resume:
            last = torch.load(
                run_dir / "last.pt",
                map_location=device,
                weights_only=False,
            )
            model.load_state_dict(last["model"])
            optimizer.load_state_dict(last["optimizer"])

            completed = last["epoch"]
            best = last["best_valid_l2"]
            best_epoch = last["best_epoch"]
            history = last["history"]

            state = last["rng"]
            random.setstate(state["python"])
            np.random.set_state(state["numpy"])
            torch.set_rng_state(state["torch"].cpu())

            if "cuda" in state:
                torch.cuda.set_rng_state_all([
                    value.cpu() for value in state["cuda"]
                ])

            print(
                "恢复已完成epoch：", completed,
                flush=True,
            )

        report.update(
            completed_epoch=completed,
            best_epoch=best_epoch,
            best_valid_l2=(
                best if math.isfinite(best) else None
            ),
        )

        train = parts["train"]["states"][
            :, :cfg["train_frames"]
        ]

        for epoch in range(
            completed + 1, cfg["epochs"] + 1
        ):
            model.train()
            loss_sum = 0.0
            clock = time.perf_counter()

            for _ in range(cfg["updates_per_epoch"]):
                indices = torch.randint(
                    len(train), (cfg["batch_size"],)
                )
                time_indices = torch.randint(
                    cfg["train_frames"] - 1,
                    (cfg["batch_size"],),
                )

                x = (
                    train[indices, time_indices].to(device) - sm
                ) / ss
                y = (
                    train[indices, time_indices + 1].to(device) - sm
                ) / ss
                force = (
                    parts["train"]["forces"][indices].to(device) - fm
                ) / fs

                optimizer.zero_grad(set_to_none=True)

                prediction = forward_ffno(
                    model, x, force, grid
                )
                loss = (prediction - y).square().mean()

                if not torch.isfinite(loss):
                    raise RuntimeError("训练损失NaN/Inf")

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg["grad_clip"]
                )
                optimizer.step()
                loss_sum += loss.item()

            curves = evaluate(
                model, parts["valid"], norms,
                grid, 100, times,
            )

            row = {
                "epoch": epoch,
                "train_mse": (
                    loss_sum / cfg["updates_per_epoch"]
                ),
            }

            for horizon in cfg["validation_steps"]:
                metrics = window_metrics(
                    curves, horizon
                )
                row[f"valid_{horizon}_l2"] = (
                    metrics["relative_l2"]
                )
                row[f"valid_{horizon}_velocity_l2"] = (
                    metrics["velocity_relative_l2"]
                )

            torch.cuda.synchronize()
            row["seconds"] = time.perf_counter() - clock

            improved = row["valid_100_l2"] < best
            if improved:
                best = row["valid_100_l2"]
                best_epoch = epoch

            history.append(row)

            payload = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "config": cfg,
                "split": info["split_identities"],
                "state_mean": norms[0],
                "state_std": norms[1],
                "force_mean": norms[2],
                "force_std": norms[3],
                "best_valid_l2": best,
                "best_epoch": best_epoch,
                "history": history,
                "rng": rng_state(),
            }

            if improved:
                atomic_save(
                    run_dir / "best.pt", payload
                )
            atomic_save(
                run_dir / "last.pt", payload
            )

            with (run_dir / "history.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle_csv:
                writer = csv.DictWriter(
                    handle_csv,
                    fieldnames=list(row),
                )
                writer.writeheader()
                writer.writerows(history)

            report.update(
                completed_epoch=epoch,
                best_epoch=best_epoch,
                best_valid_l2=best,
                seconds=time.perf_counter() - start_clock,
            )
            save_summary(run_dir, report)

            print(
                f"Epoch {epoch}/{cfg['epochs']} "
                f"| train MSE {row['train_mse']:.5g} "
                f"| valid L2 1/10/100: "
                f"{row['valid_1_l2']:.5g}/"
                f"{row['valid_10_l2']:.5g}/"
                f"{row['valid_100_l2']:.5g}",
                flush=True,
            )

        best_checkpoint = torch.load(
            run_dir / "best.pt",
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(
            best_checkpoint["model"]
        )

        curves = evaluate(
            model, parts["test"], norms,
            grid, 200, times,
        )

        with (run_dir / "test_rollout.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle_csv:
            writer = csv.DictWriter(
                handle_csv,
                fieldnames=list(curves[0]),
            )
            writer.writeheader()
            writer.writerows(curves)

        report["test"] = {
            str(horizon): window_metrics(curves, horizon)
            for horizon in cfg["test_steps"]
        }
              
        report.update(
            status="completed",
            seconds=time.perf_counter() - start_clock,
        )
        save_summary(run_dir, report)

        # 自动画图。画图失败不影响已保存的训练和测试结果。
        try:
            save_experiment_figures(run_dir)
        except Exception:
            import traceback
            print(
                "训练和测试已完成，但画图失败，详情如下：",
                flush=True,
            )
            traceback.print_exc()

        for horizon, metrics in report["test"].items():
            print(
                f"Test {horizon}步 "
                f"| 联合L2={metrics['relative_l2']:.6f} "
                f"| 速度L2={metrics['velocity_relative_l2']:.6f}",
                flush=True,
            )

        print(
            "Summary：", run_dir / "summary.md",
            flush=True,
        )

    except BaseException as error:
        report.update(
            status="interrupted_or_failed",
            error=repr(error),
            seconds=time.perf_counter() - start_clock,
        )
        save_summary(run_dir, report)
        raise

    finally:
        sys.stdout, sys.stderr = stdout, stderr
        handle.close()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--plot":
        save_experiment_figures(
            Path(sys.argv[2]).resolve()
        )
    else:
        main()