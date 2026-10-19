"""受控外力变化：A预训练、B直接微调、B加有限A样本回放。"""
from __future__ import annotations

import argparse
import copy
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

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ns_experiment.common import RunLog
from ns_experiment.model_memory import build_ffno, forward_ffno, make_grid

CFG = {
    "seed": 0,
    "device": "cuda",
    "prediction_mode": "increment",
    "width": 32,
    "modes": 12,
    "layers": 4,
    "factor": 4,
    "batch_size": 8,
    "epochs_A": 30,
    "epochs_B": 10,
    "updates_per_epoch": 20,
    "learning_rate_A": 1e-3,
    "learning_rate_B": 1e-4,
    "lambda_replay": 1.0,
    "replay_trajectories": 2,
    "replay_entries": 64,
    "frames": 101,
    "rollout_steps": 100,
}
SPLIT = {"train": [0, 1, 2, 3], "valid": [4], "test": [5]}


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_pt(path, value):
    temporary = path.with_suffix(".pt.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data(directory):
    data = {"A": {}, "B": {}}
    identities = []
    pair_ids = {}
    expected_time = torch.arange(101, dtype=torch.float64) * 0.005

    for group, indices in SPLIT.items():
        collected = {"A": [], "B": []}
        for index in indices:
            paired = {}
            for task in ("A", "B"):
                path = directory / f"task_{task}_pair_{index:03d}.pt"
                if not path.exists():
                    raise FileNotFoundError(f"数据尚未齐全：{path}")

                payload = torch.load(
                    path, map_location="cpu", weights_only=False
                )
                state = payload["states"].float()
                force = payload["forces"].float()
                times = payload["times"].double()

                if state.shape != (1, 101, 64, 64, 3):
                    raise RuntimeError(f"状态形状错误：{path}")
                if force.shape != (1, 64, 64, 2):
                    raise RuntimeError(f"外力形状错误：{path}")
                if times.shape != (1, 101):
                    raise RuntimeError(f"时间形状错误：{path}")
                if not torch.allclose(times[0], expected_time):
                    raise RuntimeError(f"时间坐标不符：{path}")
                if payload["task"] != task:
                    raise RuntimeError(f"任务标签不符：{path}")
                if payload["config"].get("NU") != 0.01:
                    raise RuntimeError(f"黏性系数不符：{path}")
                if not (
                    torch.isfinite(state).all()
                    and torch.isfinite(force).all()
                ):
                    raise RuntimeError(f"存在非有限值：{path}")

                paired[task] = (state, force, str(payload["pair_id"]))
                collected[task].append((state, force))
                identities.append({
                    "path": str(path.resolve()),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "task": task,
                    "pair_index": index,
                    "pair_id": str(payload["pair_id"]),
                    "experiment_split": group,
                    "source_split": payload.get("split"),
                    "source_config": payload["config"],
                    "max_divergence_rms": payload.get("max_divergence_rms"),
                })

            a, b = paired["A"], paired["B"]
            if a[2] != b[2]:
                raise RuntimeError("A/B配对身份不同。")
            if a[2] in pair_ids:
                raise RuntimeError("存在重复轨迹身份，可能造成划分泄漏。")
            pair_ids[a[2]] = group
            if not torch.allclose(a[0][:, 0], b[0][:, 0], atol=1e-6, rtol=1e-5):
                raise RuntimeError("A/B初态不一致。")
            if not torch.allclose(b[1], 2 * a[1], atol=1e-6, rtol=1e-5):
                raise RuntimeError("B外力不是A外力的两倍。")

        for task in ("A", "B"):
            data[task][group] = {
                "states": torch.cat([v[0] for v in collected[task]]),
                "forces": torch.cat([v[1] for v in collected[task]]),
            }
    return data, identities


def normalizers(data):
    states = data["A"]["train"]["states"]
    forces = data["A"]["train"]["forces"]
    return {
        "state_mean": states.mean(dim=(0, 1, 2, 3)),
        "state_std": states.std(
            dim=(0, 1, 2, 3), unbiased=False
        ).clamp_min(1e-6),
        "force_mean": forces.mean(dim=(0, 1, 2)),
        "force_std": forces.std(
            dim=(0, 1, 2), unbiased=False
        ).clamp_min(1e-6),
    }


def normalized(part, norms):
    return {
        "states": (
            part["states"] - norms["state_mean"]
        ) / norms["state_std"],
        "forces": (
            part["forces"] - norms["force_mean"]
        ) / norms["force_std"],
    }


def sample_batch(part, generator, device):
    states = part["states"]
    owners = torch.randint(
        len(states), (CFG["batch_size"],), generator=generator
    )
    times = torch.randint(
        100, (CFG["batch_size"],), generator=generator
    )
    return (
        states[owners, times].to(device),
        part["forces"][owners].to(device),
        states[owners, times + 1].to(device),
    )


def make_buffer(part):
    generator = torch.Generator().manual_seed(2026)
    owners = torch.randperm(
        len(part["states"]), generator=generator
    )[:CFG["replay_trajectories"]]
    chosen = torch.randperm(
        len(owners) * 100, generator=generator
    )[:CFG["replay_entries"]]
    selected_owner = owners[chosen // 100]
    selected_time = chosen % 100
    return {
        "state": part["states"][selected_owner, selected_time].clone(),
        "force": part["forces"][selected_owner].clone(),
        "target": part["states"][selected_owner, selected_time + 1].clone(),
        "owner_local_index": selected_owner.clone(),
        "time_index": selected_time.clone(),
        "source_pair_indices": [
            SPLIT["train"][int(i)] for i in owners
        ],
    }


def sample_buffer(buffer, generator, device):
    indices = torch.randint(
        len(buffer["state"]),
        (CFG["batch_size"],),
        generator=generator,
    )
    return tuple(
        buffer[key][indices].to(device)
        for key in ("state", "force", "target")
    )


def update(model, optimizer, batch, grid, old_batch=None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    state, force, target = batch
    loss_new = (forward_ffno(model, state, force, grid) - target).square().mean()
    loss_old = torch.zeros((), device=state.device)
    if old_batch is not None:
        old_state, old_force, old_target = old_batch
        loss_old = (
            forward_ffno(model, old_state, old_force, grid) - old_target
        ).square().mean()
    loss = loss_new + CFG["lambda_replay"] * loss_old
    if not torch.isfinite(loss):
        raise RuntimeError("训练损失出现NaN/Inf。")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), 1.0, error_if_nonfinite=True
    )
    optimizer.step()
    return float(loss_new.detach()), float(loss_old.detach())


def relative_l2(prediction, truth):
    difference = (prediction - truth).double().flatten(1)
    target = truth.double().flatten(1)
    return (
        difference.norm(dim=1)
        / target.norm(dim=1).clamp_min(1e-12)
    ).mean().item()


@torch.no_grad()
def evaluate(model, part, norms, grid, device, field_prefix=None):
    model.eval()
    truth = part["states"].to(device)
    mean = norms["state_mean"].to(device)
    std = norms["state_std"].to(device)
    force = (
        (part["forces"] - norms["force_mean"])
        / norms["force_std"]
    ).to(device)
    current = (truth[:, 0] - mean) / std
    persistence = truth[:, 0]
    snapshots = []
    rows = []

    for step in range(1, 101):
        current = forward_ffno(model, current, force, grid)
        prediction = current * std + mean
        target = truth[:, step]
        if not torch.isfinite(prediction).all():
            return {
                "finite": False,
                "relative_l2": float("inf"),
                "velocity_relative_l2": float("inf"),
                "steps": rows,
            }
        if field_prefix is not None and step in (1, 10, 50, 100):
            snapshots.append((
                step,
                prediction[0].detach().cpu().numpy().copy(),
                target[0].detach().cpu().numpy().copy(),
            ))
        rows.append({
            "step": step,
            "time": step * 0.005,
            "persistence_relative_l2": relative_l2(persistence, target),
            "persistence_velocity_relative_l2": relative_l2(
                persistence[..., :2], target[..., :2]
            ),
            "relative_l2": relative_l2(prediction, target),
            "velocity_relative_l2": relative_l2(
                prediction[..., :2], target[..., :2]
            ),
            "particles_relative_l2": relative_l2(
                prediction[..., 2:], target[..., 2:]
            ),
            "mse": (
                (prediction - target).double().square().mean().item()
            ),
        })
    if field_prefix is not None:
        save_field_comparison(field_prefix, snapshots)
    return {
        "finite": True,
        "horizons": horizon_rows(rows),
        "persistence_relative_l2": float(np.mean([
            r["persistence_relative_l2"] for r in rows
        ])),
        "persistence_velocity_relative_l2": float(np.mean([
            r["persistence_velocity_relative_l2"] for r in rows
        ])),
        "relative_l2": float(np.mean([r["relative_l2"] for r in rows])),
        "velocity_relative_l2": float(np.mean([
            r["velocity_relative_l2"] for r in rows
        ])),
        "final_relative_l2": rows[-1]["relative_l2"],
        "mse": float(np.mean([r["mse"] for r in rows])),
        "steps": rows,
    }


def checkpoint(model, optimizer, epoch, norms, history):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "config": CFG,
        "split": SPLIT,
        **norms,
        "history": history,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
        },
    }


def plot_results(directory, histories, tests):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, stage in zip(axes, ("pretrain_A", "finetune", "replay")):
        rows = histories[stage]
        for task in ("A", "B"):
            key = f"valid_{task}"
            values = [r.get(key, float("nan")) for r in rows]
            values = [v if math.isfinite(v) else np.nan for v in values]
            ax.plot([r["epoch"] for r in rows], values, label=f"valid {task}")
        ax.set_title(stage)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean rollout relative L2")
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "validation_curves.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for column, task in enumerate(("A", "B")):
        for stage, values in tests.items():
            rows = values[task]["steps"]
            if stage == "before_B":
                for row_index, key in enumerate((
                    "persistence_relative_l2",
                    "persistence_velocity_relative_l2",
                )):
                    axes[row_index, column].plot(
                        [r["time"] for r in rows],
                        [r[key] for r in rows],
                        "--", color="black", label="Persistence",
                    )
            axes[0, column].plot(
                [r["time"] for r in rows],
                [r["relative_l2"] for r in rows], label=stage,
            )
            axes[1, column].plot(
                [r["time"] for r in rows],
                [r["velocity_relative_l2"] for r in rows], label=stage,
            )
        for row, metric in enumerate(("Joint relative L2", "Velocity relative L2")):
            ax = axes[row, column]
            ax.set_title(f"Task {task}: {metric}")
            ax.set_xlabel("Physical time")
            ax.set_yscale("symlog", linthresh=1e-3)
            ax.grid(alpha=0.25)
            ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "test_rollout_curves.png", dpi=180)
    plt.close(fig)



def save_field_comparison(prefix, snapshots):
    for quantity in ("speed", "particles"):
        truths, predictions = [], []
        for step, prediction, truth in snapshots:
            if quantity == "speed":
                truths.append(np.linalg.norm(truth[..., :2], axis=-1))
                predictions.append(
                    np.linalg.norm(prediction[..., :2], axis=-1)
                )
            else:
                truths.append(truth[..., 2])
                predictions.append(prediction[..., 2])

        errors = [
            np.abs(p - t) for p, t in zip(predictions, truths)
        ]
        low = min(float(a.min()) for a in truths + predictions)
        high = max(float(a.max()) for a in truths + predictions)
        high = max(high, low + 1e-12)
        error_max = max(max(float(a.max()) for a in errors), 1e-12)

        fig, axes = plt.subplots(
            3, len(snapshots), figsize=(13, 8),
            squeeze=False, constrained_layout=True,
        )
        for col, (step, _, _) in enumerate(snapshots):
            for row, field in enumerate(
                (truths[col], predictions[col], errors[col])
            ):
                ax = axes[row, col]
                image = ax.imshow(
                    field.T, origin="lower", extent=(0, 1, 0, 1),
                    cmap="magma" if row == 2 else "viridis",
                    vmin=0 if row == 2 else low,
                    vmax=error_max if row == 2 else high,
                )
                ax.set_title(f"step {step}, t={step * 0.005:.3f}")
                if col == 0:
                    ax.set_ylabel(
                        ("Truth", "Prediction", "Absolute error")[row]
                    )
                ax.set_xticks([])
                ax.set_yticks([])
                fig.colorbar(image, ax=ax, shrink=0.75)
        fig.suptitle(f"{prefix.name}: {quantity}")
        fig.savefig(
            prefix.parent / f"{prefix.name}_{quantity}.png", dpi=160
        )
        plt.close(fig)


def horizon_rows(rows):
    reports = []
    for horizon in (1, 10, 50, 100):
        selected = rows[:horizon]
        if len(selected) != horizon:
            continue
        reports.append({
            "horizon": horizon,
            "time": horizon * 0.005,
            "mean_relative_l2": float(np.mean([
                r["relative_l2"] for r in selected
            ])),
            "endpoint_relative_l2": selected[-1]["relative_l2"],
            "mean_velocity_relative_l2": float(np.mean([
                r["velocity_relative_l2"] for r in selected
            ])),
            "endpoint_velocity_relative_l2":
                selected[-1]["velocity_relative_l2"],
            "mean_particles_relative_l2": float(np.mean([
                r["particles_relative_l2"] for r in selected
            ])),
            "persistence_mean_relative_l2": float(np.mean([
                r["persistence_relative_l2"] for r in selected
            ])),
            "persistence_mean_velocity_relative_l2": float(np.mean([
                r["persistence_velocity_relative_l2"] for r in selected
            ])),
        })
    return reports



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path,
        default=ROOT / "data/cl_generated/force_shift_pilot_v1",
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("当前解释器不能使用CUDA。")
    device = torch.device("cuda")
    torch.set_num_threads(4)
    seed_all(CFG["seed"])
    data, identities = load_data(args.data_dir)
    norms = normalizers(data)
    training = {
        task: normalized(data[task]["train"], norms)
        for task in ("A", "B")
    }
    run_dir = ROOT / "results" / datetime.now().strftime(
        "ns_cl_pilot_%Y%m%d_%H%M%S_%f"
    )
    logger = RunLog(run_dir)
    start = time.perf_counter()

    try:
        write_json(run_dir / "config.json", {
            **CFG, "split": SPLIT,
            "data_dir": str(args.data_dir.resolve()),
            "check_only": args.check_only,
        })
        write_json(run_dir / "data_identity.json", identities)
        write_json(run_dir / "normalization.json", {
            k: v.tolist() for k, v in norms.items()
        })
        model = build_ffno(CFG).to(device)
        if getattr(model, "prediction_mode", None) != "increment":
            raise RuntimeError(
                "model_memory.py尚未支持increment，请先检查build_ffno。"
            )
        grid = make_grid(64, 64, device)
        print("结果目录：", run_dir, flush=True)
        print("GPU：", torch.cuda.get_device_name(0), flush=True)
        print("每工况train/valid/test：4/1/1；滚动100步。", flush=True)
        print("预测方式：increment；归一化仅使用A训练集。", flush=True)

        if args.check_only:
            generator = torch.Generator().manual_seed(123)
            batch = sample_batch(training["A"], generator, device)
            model.train()
            prediction = forward_ffno(model, batch[0], batch[1], grid)
            if prediction.shape != batch[2].shape:
                raise RuntimeError("输出形状错误。")
            loss = (prediction - batch[2]).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("检查损失非有限。")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            torch.cuda.synchronize()
            write_json(run_dir / "summary.json", {
                "status": "check_passed_no_training",
                "loss": float(loss.detach()),
            })
            (run_dir / "summary.md").write_text(
                "# CL训练入口检查\n\n"
                "- 数据、CUDA前后向通过。\n"
                "- 未执行参数更新；不是实验精度结果。\n",
                encoding="utf-8",
            )
            print("CHECK PASSED：数据和CUDA前后向通过，未训练。", flush=True)
            return

        histories = {"pretrain_A": [], "finetune": [], "replay": []}
        optimizer = torch.optim.Adam(
            model.parameters(), lr=CFG["learning_rate_A"]
        )
        generator = torch.Generator().manual_seed(123)
        best_valid = float("inf")
        best_epoch = None

        for epoch in range(1, CFG["epochs_A"] + 1):
            losses = []
            for _ in range(CFG["updates_per_epoch"]):
                batch = sample_batch(training["A"], generator, device)
                losses.append(update(model, optimizer, batch, grid)[0])
            valid = evaluate(model, data["A"]["valid"], norms, grid, device)
            row = {
                "epoch": epoch,
                "train_mse": float(np.mean(losses)),
                "valid_A": valid["relative_l2"],
            }
            histories["pretrain_A"].append(row)
            state = checkpoint(model, optimizer, epoch, norms, histories["pretrain_A"])
            save_pt(run_dir / "A_last.pt", state)
            if valid["relative_l2"] < best_valid:
                best_valid = valid["relative_l2"]
                best_epoch = epoch
                save_pt(run_dir / "A_best.pt", state)
            save_csv(run_dir / "pretrain_A_history.csv", histories["pretrain_A"])
            print(
                f"A预训练 {epoch}/{CFG['epochs_A']} | "
                f"MSE={row['train_mse']:.6g} | "
                f"valid A={row['valid_A']:.6g}",
                flush=True,
            )

        if best_epoch is None:
            raise RuntimeError("A验证滚动始终非有限，尚不能进入CL对照。")

        best = torch.load(
            run_dir / "A_best.pt", map_location="cpu", weights_only=False
        )
        model.load_state_dict(best["model"])
        initial_weights = copy.deepcopy(best["model"])
        buffer = make_buffer(training["A"])
        save_pt(run_dir / "replay_buffer.pt", buffer)
        tests = {}

        def test_stage(stage):
            results = {}
            for task in ("A", "B"):
                result = evaluate(
                    model, data[task]["test"], norms, grid, device,
                    field_prefix=run_dir / f"{stage}_test_{task}",
                )
                if not result["finite"]:
                    raise RuntimeError(f"{stage}在工况{task}测试时滚动非有限。")
                save_csv(
                    run_dir / f"{stage}_test_{task}.csv", result["steps"]
                )
                save_csv(
                    run_dir / f"{stage}_test_{task}_horizons.csv",
                    result["horizons"],
                )
                results[task] = result
                print(
                    f"{stage} | test {task} | "
                    f"联合L2={result['relative_l2']:.6f} | "
                    f"速度L2={result['velocity_relative_l2']:.6f}",
                    flush=True,
                )
            return results

        tests["before_B"] = test_stage("before_B")

        for stage in ("finetune", "replay"):
            model.load_state_dict(initial_weights)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=CFG["learning_rate_B"]
            )
            new_generator = torch.Generator().manual_seed(456)
            old_generator = torch.Generator().manual_seed(789)

            for epoch in range(1, CFG["epochs_B"] + 1):
                new_losses, old_losses = [], []
                for _ in range(CFG["updates_per_epoch"]):
                    batch = sample_batch(
                        training["B"], new_generator, device
                    )
                    old_batch = (
                        sample_buffer(buffer, old_generator, device)
                        if stage == "replay" else None
                    )
                    new_loss, old_loss = update(
                        model, optimizer, batch, grid, old_batch
                    )
                    new_losses.append(new_loss)
                    old_losses.append(old_loss)

                va = evaluate(model, data["A"]["valid"], norms, grid, device)
                vb = evaluate(model, data["B"]["valid"], norms, grid, device)
                row = {
                    "epoch": epoch,
                    "new_mse": float(np.mean(new_losses)),
                    "old_mse": float(np.mean(old_losses)),
                    "valid_A": va["relative_l2"],
                    "valid_B": vb["relative_l2"],
                }
                histories[stage].append(row)
                save_pt(
                    run_dir / f"{stage}_last.pt",
                    checkpoint(model, optimizer, epoch, norms, histories[stage]),
                )
                save_csv(run_dir / f"{stage}_history.csv", histories[stage])
                print(
                    f"{stage} {epoch}/{CFG['epochs_B']} | "
                    f"valid A={row['valid_A']:.6g} | "
                    f"valid B={row['valid_B']:.6g}",
                    flush=True,
                )

            tests[stage] = test_stage(stage)

        torch.cuda.synchronize()
        compact = {
            stage: {
                task: {k: v for k, v in result.items() if k != "steps"}
                for task, result in values.items()
            }
            for stage, values in tests.items()
        }
        changes = {
            stage: compact[stage]["A"]["relative_l2"]
            - compact["before_B"]["A"]["relative_l2"]
            for stage in ("finetune", "replay")
        }
        report = {
            "status": "completed",
            "config": CFG,
            "split": SPLIT,
            "best_A_epoch": best_epoch,
            "best_A_valid_relative_l2": best_valid,
            "test": compact,
            "old_task_error_change": changes,
            "replay_source_pair_indices": buffer["source_pair_indices"],
            "replay_entries": len(buffer["state"]),
            "memory_bank_used": False,
            "total_seconds": time.perf_counter() - start,
            "limitations": [
                "单种子，每工况只有一条验证和一条测试轨迹。",
                "自生成受控外力变化，不是PDEBench精确复现。",
                "回放与微调看到相同B样本，回放额外使用A样本，计算量不相等。",
                "A预训练按A验证误差选检查点；适应后使用固定训练预算的末轮。",
                "没有残差记忆修正，也没有物理残差损失。",
                "last检查点已保存，但本脚本没有自动恢复训练入口。",
            ],
        }
        write_json(run_dir / "summary.json", report)
        plot_results(run_dir, histories, tests)

        lines = [
            "# NS 外力变化持续学习小规模实验", "",
            "## 配置", "",
            "- 自生成A/B工况，B外力为A的两倍；同一pair共享初态。",
            "- 每工况4条训练、1条验证、1条测试，pair之间不交叉。",
            "- 64×64，101帧，保存间隔0.005；滚动100步到t=0.5。",
            "- FFNO增量预测；输入速度、particles、外力和坐标。",
            "- 归一化仅由A训练数据计算，后续保持不变。",
            f"- A训练30×20次更新，最佳epoch={best_epoch}。",
            "- 两种适应均从同一A检查点出发，各10×20次更新。",
            "- A学习率1e-3，适应学习率1e-4，batch=8。",
            "- 回放从2条A训练轨迹固定保存64个转移；每次用8个B和8个旧样本。",
            "- 回放损失：B单步MSE + 1.0 × A回放单步MSE。",
            "- 本次未加入残差memory bank。", "",
            "## 测试结果", "",
            "| 方法 | A联合L2 | B联合L2 | A速度L2 | B速度L2 |",
            "|---|---:|---:|---:|---:|",
        ]
        for stage, values in compact.items():
            a, b = values["A"], values["B"]
            lines.append(
                f"| {stage} | {a['relative_l2']:.6f} | "
                f"{b['relative_l2']:.6f} | "
                f"{a['velocity_relative_l2']:.6f} | "
                f"{b['velocity_relative_l2']:.6f} |"
            )
        lines += [
            "", "误差为各轨迹、各预测步Relative L2的平均。",
            "", "## 旧工况误差变化", "",
            f"- 直接微调：{changes['finetune']:+.6f}",
            f"- 回放更新：{changes['replay']:+.6f}",
            "- 相对适应前，正值表示A变差，负值表示A改善。",
            "", "## 图像", "",
            "![验证误差](validation_curves.png)", "",
            "![A/B测试滚动误差](test_rollout_curves.png)", "",
            "## 局限", "",
        ]
        lines += [f"- {v}" for v in report["limitations"]]

        lines += [
            "", "## 保持初态不变的对照", "",
            "Persistence将初态作为所有未来时刻的预测。",
            "",
            "| 工况 | Persistence联合L2 | Persistence速度L2 |",
            "|---|---:|---:|",
        ]
        for task in ("A", "B"):
            value = compact["before_B"][task]
            lines.append(
                f"| {task} | {value['persistence_relative_l2']:.6f} | "
                f"{value['persistence_velocity_relative_l2']:.6f} |"
            )

        lines += [
            "", "## 不同滚动长度", "",
            "平均误差为前H步平均；末步误差为恰好第H步的误差。",
            "",
            "| 方法 | 工况 | H | 平均联合L2 | 末步联合L2 | 平均速度L2 |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for stage, values in compact.items():
            for task in ("A", "B"):
                for row in values[task]["horizons"]:
                    lines.append(
                        f"| {stage} | {task} | {row['horizon']} | "
                        f"{row['mean_relative_l2']:.6f} | "
                        f"{row['endpoint_relative_l2']:.6f} | "
                        f"{row['mean_velocity_relative_l2']:.6f} |"
                    )

        lines += [
            "", "## 测试场对比", "",
            "展示每个工况第一条测试轨迹的第1、10、50、100步。",
            "从上到下为真值、预测、绝对误差。",
            "每张图真值与预测共用色标；误差使用独立色标。",
            "速度图展示速度大小，不展示方向。",
        ]
        for stage in tests:
            for task in ("A", "B"):
                prefix = f"{stage}_test_{task}"
                lines += [
                    "", f"### {stage} — 工况 {task}", "",
                    f"![速度大小]({prefix}_speed.png)", "",
                    f"![Particles]({prefix}_particles.png)",
                ]

        (run_dir / "summary.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        print("实验完成，Summary：", run_dir / "summary.md", flush=True)

    except BaseException as error:
        write_json(run_dir / "failure.json", {
            "type": type(error).__name__, "message": str(error)
        })
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    main()
