"""当前多轨迹 NS 的固定记忆与可训练多步记忆实验。"""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "4"

import csv
import json
import math
import sys
import time
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

from ns_experiment.common import (
    RunLog, seed_everything, write_json, rng_state,
)
from ns_experiment.run_baseline_cohort import (
    prepare, window_metrics, atomic_save,
)
from ns_experiment.model_memory import (
    MemoryBank,
    CandidateScorer,
    build_ffno,
    make_grid,
    freeze_module,
    build_memory_bank,
    state_force_features,
    residual_stats,
    fixed_delta,
    learned_delta,
    retrieve,
    rollout,
    evaluate_autoregressive,
    assert_later_ffno_sees_correction,
)

BASELINE_DIR = (
    ROOT / "results"
    / "ns_cohort_B_20261018_111645_170902"
)

CFG = {
    "seed": 0,
    "device": "cuda",
    "bank_entries": 256,
    "top_k": 3,
    "temperature": 1.0,
    "gamma_candidates": [0.0, 0.25, 0.5, 1.0],
    "router_epochs": 10,
    "updates_per_epoch": 20,
    "batch_size": 4,
    "horizon": 3,
    "learning_rate": 1e-3,
    "validation_steps": 100,
    "test_steps": 200,
}


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_corrector(kind, bank, scorer, gamma):
    if gamma == 0:
        return None

    def corrector(current, force, owners):
        if kind == "fixed":
            return fixed_delta(
                bank, current, force, owners,
                CFG["top_k"], CFG["temperature"], gamma,
            )
        return learned_delta(
            scorer, bank, current, force, owners,
            CFG["top_k"], gamma,
        )

    return corrector


def evaluate(model, part, norms, grid, times, steps, corrector=None):
    result = evaluate_autoregressive(
        model,
        part["states"],
        part["forces"],
        *norms,
        grid,
        steps,
        part["indices"],
        corrector,
    )
    for row in result["steps"]:
        if not all(math.isfinite(float(v)) for v in row.values()):
            raise FloatingPointError("滚动预测指标出现 NaN/Inf")
        row["time"] = float(times[int(row["step"])])
    return result["steps"]


def validation_score(model, part, norms, grid, times, corrector):
    try:
        curves = evaluate(
            model, part, norms, grid, times,
            CFG["validation_steps"], corrector,
        )
    except FloatingPointError:
        return None
    return window_metrics(
        curves, CFG["validation_steps"]
    )["relative_l2"]


def search_gamma(kind, model, parts, norms, grid, times,
                 bank, scorer, baseline_valid, output):
    rows = []
    # gamma=0 是无修正候选；相同分数时优先保留较小 gamma。
    selected_gamma = 0.0
    selected_score = baseline_valid

    for gamma in CFG["gamma_candidates"]:
        score = (
            baseline_valid if gamma == 0 else
            validation_score(
                model, parts["valid"], norms, grid, times,
                make_corrector(kind, bank, scorer, gamma),
            )
        )
        rows.append({
            "method": kind,
            "gamma": gamma,
            "valid_l2": score,
            "finite": score is not None,
        })
        save_csv(output / f"gamma_search_{kind}.csv", rows)
        print(
            f"验证 {kind}: gamma={gamma}, L2={score}",
            flush=True,
        )
        if score is not None and score < selected_score:
            selected_gamma, selected_score = gamma, score

    return selected_gamma, selected_score


def save_figures(output, history, curves):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs = [row["epoch"] for row in history]

    axes[0].plot(epochs, [row["train_mse"] for row in history])
    axes[0].set(
        title="Memory router training",
        xlabel="Epoch", ylabel="Normalized multi-step MSE",
    )
    axes[0].set_yscale("symlog", linthresh=1e-8)

    values = [
        np.nan if row["valid_l2_gamma1"] is None
        else row["valid_l2_gamma1"]
        for row in history
    ]
    axes[1].plot(epochs, values)
    axes[1].set(
        title="Validation rollout with gamma=1",
        xlabel="Epoch", ylabel="Mean 100-step Relative L2",
    )
    axes[1].set_yscale("symlog", linthresh=1e-4)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "router_training.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for method, rows in curves.items():
        if rows is None:
            continue
        times = [row["time"] for row in rows]
        axes[0].plot(
            times, [row["relative_l2"] for row in rows],
            label=method,
        )
        axes[1].plot(
            times, [row["velocity_relative_l2"] for row in rows],
            label=method,
        )
    for ax, title in zip(axes, ("Joint state", "Velocity vector")):
        ax.axvline(0.5, color="gray", linestyle="--")
        ax.set(
            title=title, xlabel="Physical time",
            ylabel="Per-step Relative L2",
        )
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "test_comparison.png", dpi=200)
    plt.close(fig)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("请在 AutoDL 的 CUDA 环境运行。")

    baseline_dir = (
        Path(sys.argv[1]).resolve()
        if len(sys.argv) > 1 else BASELINE_DIR
    )
    checkpoint = torch.load(
        baseline_dir / "best.pt",
        map_location="cpu", weights_only=False,
    )
    baseline_cfg = checkpoint["config"]

    output = ROOT / "results" / datetime.now().strftime(
        "ns_memory_cohort_%Y%m%d_%H%M%S_%f"
    )
    logger = RunLog(output)
    start_clock = time.perf_counter()
    report = {
        "status": "running",
        "config": CFG,
        "baseline_dir": str(baseline_dir),
        "baseline_epoch": int(checkpoint["epoch"]),
    }

    try:
        print("结果目录：", output, flush=True)
        print("GPU：", torch.cuda.get_device_name(0), flush=True)
        print("主干 epoch：", checkpoint["epoch"], flush=True)

        torch.set_num_threads(4)
        device = torch.device("cuda")

        parts, computed_norms, times, info = prepare(baseline_cfg)
        old_info = json.loads(
            (baseline_dir / "data_identity.json").read_text(
                encoding="utf-8"
            )
        )
        for key in ("cache_sha256", "split_identities"):
            if info[key] != old_info[key]:
                raise RuntimeError(f"数据或划分变化：{key}")
        if checkpoint["split"] != info["split_identities"]:
            raise RuntimeError("检查点轨迹划分不一致。")

        names = (
            "state_mean", "state_std",
            "force_mean", "force_std",
        )
        norms = tuple(
            checkpoint[name].detach().cpu().float()
            for name in names
        )
        for name, saved, computed in zip(
            names, norms, computed_norms
        ):
            if not torch.allclose(
                saved, computed.cpu().float(),
                rtol=1e-5, atol=1e-6,
            ):
                raise RuntimeError(f"归一化不一致：{name}")

        write_json(output / "config.json", {
            "memory": CFG, "baseline": baseline_cfg,
        })
        write_json(output / "data_identity.json", info)
        # 固定这次实验实际使用的检查点。
        torch.save(checkpoint, output / "source_checkpoint.pt")

        seed_everything(CFG["seed"])
        model = build_ffno(baseline_cfg).to(device)
        model.load_state_dict(checkpoint["model"])
        freeze_module(model)
        original_weights = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

        frames = int(baseline_cfg["train_frames"])
        mean, std, force_mean, force_std = norms
        train_states = (
            parts["train"]["states"][:, :frames] - mean
        ) / std
        train_forces = (
            parts["train"]["forces"] - force_mean
        ) / force_std

        height, width = train_states.shape[2:4]
        grid = make_grid(height, width, device)
        train_ids = parts["train"]["indices"]
        print(
            f"划分：train={len(train_ids)}, "
            f"valid={len(parts['valid']['indices'])}, "
            f"test={len(parts['test']['indices'])}",
            flush=True,
        )

        assert_later_ffno_sees_correction(
            model,
            train_states[:1, 0].to(device),
            train_forces[:1].to(device),
            grid,
            train_states[:1, 1].to(device),
        )
        print("跨步 FFNO 输入梯度检查通过。", flush=True)

        bank = build_memory_bank(
            model, train_states, train_forces, grid,
            train_ids, CFG["bank_entries"], device, CFG["seed"],
        )
        bank.save(output / "memory_bank.pt")
        # 使用磁盘上的 float16 残差版本，保证重载一致。
        bank = MemoryBank.load(output / "memory_bank.pt", device)

        if bank.trajectory_ids.unique().numel() < 2:
            raise RuntimeError("记忆库覆盖的训练轨迹不足两条。")
        chosen, _ = retrieve(
            bank, bank.keys[0],
            int(bank.trajectory_ids[0]), CFG["top_k"],
        )
        if chosen.numel() == 0 or torch.any(
            bank.trajectory_ids[chosen] == bank.trajectory_ids[0]
        ).item():
            raise RuntimeError("排除自身轨迹检查失败。")

        feature_dim = int(bank.keys.shape[-1])
        stat_dim = residual_stats(
            torch.zeros(height, width, 3, device=device)
        ).numel()
        scorer = CandidateScorer(feature_dim, stat_dim).to(device)
        optimizer = torch.optim.Adam(
            scorer.parameters(), lr=CFG["learning_rate"]
        )
        report.update(
            bank_entries=bank.size,
            feature_dim=feature_dim,
            router_parameters=sum(
                p.numel() for p in scorer.parameters()
            ),
        )
        write_json(output / "summary.json", report)
        print(
            f"记忆条目：{bank.size}，特征维：{feature_dim}",
            flush=True,
        )

        base_valid = validation_score(
            model, parts["valid"], norms, grid, times, None
        )
        if base_valid is None:
            raise RuntimeError("主干验证结果非有限值。")

        history = []
        best_score = float("inf")
        best_epoch = None
        horizon = CFG["horizon"]

        for epoch in range(1, CFG["router_epochs"] + 1):
            scorer.train()
            loss_sum = 0.0

            for update in range(CFG["updates_per_epoch"]):
                batch = CFG["batch_size"]
                trajectory = torch.randint(
                    train_states.shape[0], (batch,)
                )
                beginning = torch.randint(
                    train_states.shape[1] - horizon, (batch,)
                )
                initial = train_states[
                    trajectory, beginning
                ].to(device)
                force = train_forces[trajectory].to(device)
                target = torch.stack([
                    train_states[
                        trajectory, beginning + 1 + offset
                    ]
                    for offset in range(horizon)
                ], dim=1).to(device)
                owners = torch.tensor(
                    [train_ids[int(i)] for i in trajectory],
                    dtype=torch.long, device=device,
                )

                def correction(current):
                    return learned_delta(
                        scorer, bank, current, force, owners,
                        CFG["top_k"], 1.0,
                    )

                optimizer.zero_grad(set_to_none=True)
                # 这里不能包 no_grad：保留跨步输入梯度。
                prediction = rollout(
                    model, initial, force, grid,
                    horizon, correction,
                )
                loss = (prediction - target).square().mean()
                if not torch.isfinite(loss).item():
                    raise RuntimeError("修正模块训练损失非有限值。")
                loss.backward()

                gradients = [
                    p.grad for p in scorer.parameters()
                    if p.grad is not None
                ]
                if not gradients or not all(
                    torch.isfinite(g).all().item()
                    for g in gradients
                ):
                    raise RuntimeError("router 梯度缺失或非有限。")
                if not any(g.abs().sum().item() > 0 for g in gradients):
                    raise RuntimeError("router 梯度全部为零。")
                if any(p.grad is not None for p in model.parameters()):
                    raise RuntimeError("冻结主干出现参数梯度。")

                torch.nn.utils.clip_grad_norm_(
                    scorer.parameters(), 1.0
                )
                optimizer.step()
                loss_sum += loss.detach().item()

            scorer.eval()
            score = validation_score(
                model, parts["valid"], norms, grid, times,
                make_corrector("learned", bank, scorer, 1.0),
            )
            history.append({
                "epoch": epoch,
                "train_mse": loss_sum / CFG["updates_per_epoch"],
                "valid_l2_gamma1": score,
            })
            save_csv(output / "router_history.csv", history)

            payload = {
                "scorer": scorer.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "config": CFG,
                "feature_dim": feature_dim,
                "stat_dim": stat_dim,
                "history": history,
                "rng": rng_state(),
            }
            atomic_save(output / "router_last.pt", payload)
            if score is not None and score < best_score:
                best_score = score
                best_epoch = epoch
                atomic_save(output / "router_best.pt", payload)

            print(
                f"Router {epoch}/{CFG['router_epochs']} | "
                f"train MSE={history[-1]['train_mse']:.6g} | "
                f"valid gamma=1 L2={score}",
                flush=True,
            )
            write_json(output / "progress.json", {
                "completed_epoch": epoch,
                "best_epoch": best_epoch,
                "best_valid_l2_gamma1": (
                    best_score if math.isfinite(best_score) else None
                ),
            })

        fixed_gamma, fixed_score = search_gamma(
            "fixed", model, parts, norms, grid, times,
            bank, scorer, base_valid, output,
        )

        if best_epoch is not None:
            best = torch.load(
                output / "router_best.pt",
                map_location=device, weights_only=False,
            )
            scorer.load_state_dict(best["scorer"])
            scorer.eval()
            learned_gamma, learned_score = search_gamma(
                "learned", model, parts, norms, grid, times,
                bank, scorer, base_valid, output,
            )
        else:
            learned_gamma, learned_score = 0.0, base_valid
            print(
                "所有 gamma=1 的 router 验证均非有限值；"
                "学习修正回退到 gamma=0。",
                flush=True,
            )

        print("开始最终测试：B0 / B1 / B3。", flush=True)
        base_curves = evaluate(
            model, parts["test"], norms, grid, times,
            CFG["test_steps"],
        )
        curves = {"B0 FFNO": base_curves}
        test_errors = {}

        for label, kind, gamma in (
            ("B1 Fixed memory", "fixed", fixed_gamma),
            ("B3 Trained memory", "learned", learned_gamma),
        ):
            if gamma == 0:
                curves[label] = base_curves
            else:
                try:
                    curves[label] = evaluate(
                        model, parts["test"], norms, grid, times,
                        CFG["test_steps"],
                        make_corrector(kind, bank, scorer, gamma),
                    )
                except FloatingPointError as error:
                    curves[label] = None
                    test_errors[label] = str(error)

        # 确认 FFNO 权重没有被训练改变。
        if not all(
            torch.equal(value.detach().cpu(), original_weights[key])
            for key, value in model.state_dict().items()
        ):
            raise RuntimeError("FFNO 权重发生变化。")

        all_rows = []
        results = {}
        for label, rows in curves.items():
            if rows is None:
                results[label] = {"status": "nonfinite_rollout"}
                continue
            all_rows.extend([
                {"method": label, **row} for row in rows
            ])
            results[label] = {
                str(h): window_metrics(rows, h)
                for h in (1, 10, 100, 200)
            }
        save_csv(output / "test_rollout_comparison.csv", all_rows)

        report.update(
            status="completed",
            router_best_epoch=best_epoch,
            router_selection="validation 100-step L2 at gamma=1",
            selected_fixed_gamma=fixed_gamma,
            selected_learned_gamma=learned_gamma,
            validation={
                "baseline": base_valid,
                "fixed": fixed_score,
                "learned": learned_score,
            },
            test=results,
            test_errors=test_errors,
            seconds=time.perf_counter() - start_clock,
        )
        write_json(output / "summary.json", report)

        lines = [
            "# NS FFNO + 可训练残差记忆",
            "",
            f"- 主干来源：`{baseline_dir}`",
            f"- 主干 epoch：{checkpoint['epoch']}，全程冻结。",
            "- 使用原 baseline 的数据、轨迹划分和归一化。",
            f"- 记忆条目：{bank.size}，特征维：{feature_dim}。",
            f"- 记忆来自训练轨迹前 {frames - 1} 个转移。",
            "- 训练查询排除同一条轨迹的全部记忆。",
            "- 残差值由当前冻结 FFNO 重新计算。",
            "- 记忆库固定；训练 router，修正状态传入下一步。",
            "- Top-K 索引不可微；候选权重与跨步预测参与求导。",
            "- router 包含零修正候选，没有固定相似度阈值。",
            f"- 训练 H={horizon}，10轮×20次更新，batch=4。",
            f"- router 最佳 epoch：{best_epoch}。",
            f"- 固定修正 gamma：{fixed_gamma}。",
            f"- 可训练修正 gamma：{learned_gamma}。",
            "- gamma=0 表示验证集选择不使用记忆。",
            "- 验证100步选模型和gamma；测试200步不调参。",
            "- 尚未进行 CL 或主干与router联合训练。",
            "",
            "## 测试结果",
            "",
            "| 方法 | 前若干步 | 平均联合L2 | 平均速度L2 |",
            "|---|---:|---:|---:|",
        ]
        for label, values in results.items():
            if "status" in values:
                lines.append(f"| {label} | — | NaN/Inf，失败 | — |")
                continue
            for h in (1, 10, 100, 200):
                metrics = values[str(h)]
                lines.append(
                    f"| {label} | {h} "
                    f"| {metrics['relative_l2']:.6f} "
                    f"| {metrics['velocity_relative_l2']:.6f} |"
                )

        summary_path = output / "summary.md"
        summary_path.write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

        try:
            save_figures(output, history, curves)
            with summary_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    "\n## 图片\n\n"
                    "![修正模块训练](router_training.png)\n\n"
                    "测试曲线为逐步误差，虚线为训练时间窗口终点；"
                    "纵轴采用对称对数刻度。\n\n"
                    "![测试对比](test_comparison.png)\n"
                )
        except Exception:
            import traceback
            print("结果已保存，但画图失败：", flush=True)
            traceback.print_exc()

        for label, values in results.items():
            if "status" in values:
                print(label, values["status"], flush=True)
            else:
                print(
                    f"{label} | 100步平均L2="
                    f"{values['100']['relative_l2']:.6f} | "
                    f"200步平均L2="
                    f"{values['200']['relative_l2']:.6f}",
                    flush=True,
                )
        print("Summary：", summary_path, flush=True)

    except BaseException as error:
        report.update(
            status="interrupted_or_failed",
            error=repr(error),
        )
        write_json(output / "failure.json", report)
        raise
    finally:
        logger.close()



def run_cl_memory():
    import argparse
    import copy
    import hashlib
    from ns_experiment.model_memory import forward_ffno, autoregressive_fields
    from ns_experiment.run_cl import save_field_comparison

    parser = argparse.ArgumentParser()
    parser.add_argument("--cl", required=True)
    args = parser.parse_args()

    def read_json(path):
        return json.loads(path.read_text(encoding="utf-8"))

    def digest(path):
        with path.open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()

    def completed(path):
        required = [
            "summary.json", "config.json", "data_identity.json",
            "replay_buffer.pt", "finetune_last.pt", "replay_last.pt",
        ]
        return (
            all((path / name).is_file() for name in required)
            and read_json(path / "summary.json").get("status") == "completed"
        )

    if args.cl == "latest":
        choices = [
            p for p in (ROOT / "results").glob("ns_cl_pilot_*")
            if completed(p)
        ]
        if not choices:
            raise RuntimeError("尚无完成的 CL 训练。请先运行 run_cl.py。")
        source = max(choices, key=lambda p: p.name)
    else:
        source = Path(args.cl).expanduser().resolve()
        if not completed(source):
            raise RuntimeError(f"CL 结果尚未完成或文件不齐全：{source}")

    if not torch.cuda.is_available():
        raise RuntimeError("本实验需要可用的 CUDA GPU。")

    torch.set_num_threads(4)
    device = torch.device("cuda")

    config = dict(
        seed=0,
        old_entries=64,
        new_entries=192,
        top_k=3,
        horizon=3,
        epochs=10,
        updates_per_epoch=20,
        batch_size=4,
        learning_rate=1e-3,
        lambda_old=1.0,
        gamma_candidates=[0.0, 0.25, 0.5, 1.0],
        rollout_steps=100,
    )

    split = {
        "train": [0, 1, 2, 3],
        "valid": [4],
        "test": [5],
    }

    source_config = read_json(source / "config.json")
    source_summary = read_json(source / "summary.json")

    if source_config["split"] != split:
        raise RuntimeError("源 CL 实验划分与本脚本不一致。")

    data_dir = Path(source_config["data_dir"])
    identity = {
        (r["task"], int(r["pair_index"])): r
        for r in read_json(source / "data_identity.json")
    }

    # 不读取完整的 A 训练轨迹，只使用 CL 保存的有限旧样本。
    parts, data_used = {}, []

    for task in ("A", "B"):
        for group in ("train", "valid", "test"):
            if task == "A" and group == "train":
                continue

            records = []
            for pair in split[group]:
                file_path = data_dir / f"task_{task}_pair_{pair:03d}.pt"
                checksum = digest(file_path)

                if checksum != identity[(task, pair)]["sha256"]:
                    raise RuntimeError(
                        f"数据相对 CL 训练发生改变：{file_path}"
                    )

                item = torch.load(
                    file_path, map_location="cpu", weights_only=False
                )

                if item["states"].shape != (1, 101, 64, 64, 3):
                    raise RuntimeError(f"状态尺寸异常：{file_path}")
                if item["forces"].shape != (1, 64, 64, 2):
                    raise RuntimeError(f"外力尺寸异常：{file_path}")

                for key in ("states", "forces", "times"):
                    if not torch.isfinite(item[key]).all():
                        raise RuntimeError(
                            f"非有限数据：{file_path}, {key}"
                        )

                records.append(item)
                data_used.append(dict(
                    path=str(file_path),
                    sha256=checksum,
                    task=task,
                    pair=pair,
                ))

            parts[(task, group)] = {
                key: torch.cat([r[key] for r in records], dim=0)
                for key in ("states", "forces", "times")
            }

    buffer = torch.load(
        source / "replay_buffer.pt",
        map_location="cpu",
        weights_only=False,
    )

    for key, shape in (
        ("state", (64, 64, 64, 3)),
        ("force", (64, 64, 64, 2)),
        ("target", (64, 64, 64, 3)),
    ):
        if (
            tuple(buffer[key].shape) != shape
            or not torch.isfinite(buffer[key]).all()
        ):
            raise RuntimeError(f"旧样本缓存异常：{key}")

    local_owner = buffer["owner_local_index"].long()
    if (
        local_owner.shape != (64,)
        or (local_owner < 0).any()
        or (local_owner >= 4).any()
    ):
        raise RuntimeError("旧样本轨迹编号异常。")

    if len(local_owner.unique()) != 2:
        raise RuntimeError("旧样本必须来自两条训练轨迹。")

    old_owner = torch.tensor(split["train"])[local_owner]

    run_dir = ROOT / "results" / datetime.now().strftime(
        "ns_cl_memory_%Y%m%d_%H%M%S_%f"
    )
    logger = RunLog(run_dir)
    started = time.perf_counter()
    results, selections, curves = {}, {}, {}

    try:
        print("源 CL 实验：", source, flush=True)
        print("记忆实验目录：", run_dir, flush=True)

        write_json(
            run_dir / "config.json",
            dict(config=config, source=str(source), split=split),
        )
        write_json(run_dir / "data_used.json", data_used)
        torch.save(buffer, run_dir / "replay_buffer.pt")

        previous_norms = None

        for stage in ("finetune", "replay"):
            seed_everything(config["seed"])
            folder = run_dir / stage
            folder.mkdir()

            checkpoint_path = source / f"{stage}_last.pt"
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )

            if checkpoint["split"] != split:
                raise RuntimeError("检查点划分不一致。")

            norms = tuple(
                checkpoint[k].float()
                for k in (
                    "state_mean", "state_std",
                    "force_mean", "force_std",
                )
            )

            if previous_norms is not None and any(
                not torch.equal(a, b)
                for a, b in zip(norms, previous_norms)
            ):
                raise RuntimeError("两个主干归一化不一致。")

            previous_norms = norms
            sm, ss, fm, fs = [x.to(device) for x in norms]

            model = build_ffno(checkpoint["config"]).to(device)
            model.load_state_dict(checkpoint["model"])
            freeze_module(model)

            frozen = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            grid = make_grid(64, 64, device)
            torch.save(checkpoint, folder / "source_checkpoint.pt")

            train = parts[("B", "train")]
            states = (train["states"].float().to(device) - sm) / ss
            forces = (train["forces"].float().to(device) - fm) / fs

            old_state = buffer["state"].float().to(device)
            old_force = buffer["force"].float().to(device)
            old_target = buffer["target"].float().to(device)
            owners_old = old_owner.to(device)

            # B 的每条训练轨迹选48个转移，共192条。
            # 两个主干使用完全相同的抽样位置。
            generator = torch.Generator().manual_seed(2026)
            chosen_times = [
                torch.randperm(100, generator=generator)[:48]
                for _ in range(4)
            ]
            new_owner = torch.arange(4).repeat_interleave(48)
            new_time = torch.cat(chosen_times)

            new_owner_gpu = new_owner.to(device)
            new_time_gpu = new_time.to(device)

            all_state = torch.cat([
                old_state,
                states[new_owner_gpu, new_time_gpu],
            ])
            all_force = torch.cat([
                old_force,
                forces[new_owner_gpu],
            ])
            all_target = torch.cat([
                old_target,
                states[new_owner_gpu, new_time_gpu + 1],
            ])
            all_owner = torch.cat([
                owners_old,
                torch.tensor(
                    split["train"], device=device
                )[new_owner_gpu],
            ])
            all_time = torch.cat([
                buffer["time_index"].long(), new_time,
            ]).to(device)

            # 每个更新后的主干分别重算残差，避免复用过期残差。
            with torch.no_grad():
                features, residuals = [], []

                for start in range(0, len(all_state), 16):
                    s = all_state[start:start + 16]
                    f = all_force[start:start + 16]
                    y = all_target[start:start + 16]

                    features.append(state_force_features(s, f))
                    residuals.append(
                        y - forward_ffno(model, s, f, grid)
                    )

                features = torch.cat(features)
                residuals = torch.cat(residuals)

                mean = features.mean(0)
                std = features.std(
                    0, unbiased=False
                ).clamp_min(1e-6)

                bank = MemoryBank(
                    (features - mean) / std,
                    residuals,
                    all_owner,
                    all_time,
                    mean,
                    std,
                )
                bank.save(folder / "memory_bank.pt")
                bank = MemoryBank.load(
                    folder / "memory_bank.pt", device
                )

                if not torch.isfinite(bank.residuals).all():
                    raise RuntimeError("残差存储后出现非有限值。")

            write_json(
                folder / "bank_metadata.json",
                dict(
                    entries=bank.size,
                    old_entries=64,
                    new_entries=192,
                    checkpoint_sha256=digest(checkpoint_path),
                    replay_sha256=digest(
                        source / "replay_buffer.pt"
                    ),
                    old_source_pairs=buffer["source_pair_indices"],
                    exclude_same_pair_across_tasks=True,
                ),
            )

            scorer = CandidateScorer(
                bank.keys.shape[1],
                residual_stats(bank.residuals[0]).numel(),
            ).to(device)

            optimizer = torch.optim.Adam(
                scorer.parameters(),
                lr=config["learning_rate"],
            )

            def corrector(gamma):
                if gamma == 0:
                    return None
                return lambda s, f, ids: learned_delta(
                    scorer, bank, s, f, ids,
                    config["top_k"], gamma,
                )

            def evaluate(task, group, gamma):
                scorer.eval()
                part = parts[(task, group)]

                metrics = evaluate_autoregressive(
                    model,
                    part["states"],
                    part["forces"],
                    *norms,
                    grid,
                    100,
                    split[group],
                    corrector=corrector(gamma),
                )

                if not all(
                    math.isfinite(float(v))
                    for k, v in metrics.items()
                    if k != "steps"
                ):
                    return None

                for row in metrics["steps"]:
                    row["time"] = float(
                        part["times"][0, row["step"]]
                    )
                return metrics

            def validation(gamma):
                a = evaluate("A", "valid", gamma)
                b = evaluate("B", "valid", gamma)

                if a is None or b is None:
                    return None

                return (
                    a["relative_l2"] + b["relative_l2"]
                ) / 2

            baseline_valid = validation(0.0)
            if baseline_valid is None:
                raise RuntimeError(
                    "无记忆验证已出现非有限值，停止本主干实验。"
                )

            rng = torch.Generator().manual_seed(1234)
            best_loss = float("inf")
            best_weights = None
            best_epoch = None
            history = []

            for epoch in range(1, config["epochs"] + 1):
                scorer.train()
                losses = []

                for _ in range(config["updates_per_epoch"]):
                    idx = torch.randint(
                        4, (4,), generator=rng
                    ).to(device)
                    tt = torch.randint(
                        98, (4,), generator=rng
                    ).to(device)

                    ids = torch.tensor(
                        split["train"], device=device
                    )[idx]

                    current = states[idx, tt]
                    force = forces[idx]
                    predictions = []

                    # B 三步滚动：修正后的状态进入下一步 FFNO。
                    for offset in range(3):
                        current = (
                            forward_ffno(
                                model, current, force, grid
                            )
                            + learned_delta(
                                scorer, bank, current, force,
                                ids, 3, 1.0,
                            )
                        )
                        predictions.append(current)

                    target = torch.stack([
                        states[idx, tt + j + 1]
                        for j in range(3)
                    ], dim=1)

                    loss_new = (
                        torch.stack(predictions, dim=1) - target
                    ).square().mean()

                    # A 只有有限单步旧样本，不构造不存在的三步序列。
                    oi = torch.randint(
                        64, (4,), generator=rng
                    ).to(device)

                    prediction_old = (
                        forward_ffno(
                            model,
                            old_state[oi],
                            old_force[oi],
                            grid,
                        )
                        + learned_delta(
                            scorer, bank,
                            old_state[oi], old_force[oi],
                            owners_old[oi], 3, 1.0,
                        )
                    )

                    loss_old = (
                        prediction_old - old_target[oi]
                    ).square().mean()

                    loss = (
                        loss_new + config["lambda_old"] * loss_old
                    )

                    if not torch.isfinite(loss):
                        raise RuntimeError(
                            "router 训练损失非有限。"
                        )

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        scorer.parameters(),
                        1.0,
                        error_if_nonfinite=True,
                    )
                    optimizer.step()
                    losses.append(float(loss.detach()))

                valid = validation(1.0)

                history.append(dict(
                    epoch=epoch,
                    train_loss=sum(losses) / len(losses),
                    valid_mean_l2=valid,
                ))

                if valid is not None and valid < best_loss:
                    best_loss = valid
                    best_epoch = epoch
                    best_weights = copy.deepcopy(
                        scorer.state_dict()
                    )
                    torch.save(
                        dict(
                            scorer=best_weights,
                            epoch=epoch,
                            config=config,
                        ),
                        folder / "router_best.pt",
                    )

                torch.save(
                    dict(
                        scorer=scorer.state_dict(),
                        optimizer=optimizer.state_dict(),
                        epoch=epoch,
                        config=config,
                    ),
                    folder / "router_last.pt",
                )

                save_csv(
                    folder / "router_history.csv", history
                )

                print(
                    f"{stage} router {epoch}/10"
                    f" | loss={history[-1]['train_loss']:.6g}"
                    f" | valid={valid}",
                    flush=True,
                )

            search = [
                dict(
                    gamma=0.0,
                    valid_mean_l2=baseline_valid,
                )
            ]
            gamma = 0.0
            selected_valid = baseline_valid

            if best_weights is not None:
                scorer.load_state_dict(best_weights)

                for candidate in config["gamma_candidates"][1:]:
                    value = validation(candidate)
                    search.append(dict(
                        gamma=candidate,
                        valid_mean_l2=value,
                    ))

                    if (
                        value is not None
                        and value < selected_valid
                    ):
                        gamma = candidate
                        selected_valid = value

            selections[stage] = dict(
                gamma=gamma,
                best_router_epoch=best_epoch,
                search=search,
            )
            write_json(
                folder / "selection.json",
                selections[stage],
            )

            for label, strength in (
                (stage, 0.0),
                (stage + "_memory", gamma),
            ):
                results[label] = {}

                for task in ("A", "B"):
                    metrics = evaluate(
                        task, "test", strength
                    )

                    if metrics is None:
                        results[label][task] = dict(
                            status="nonfinite"
                        )
                        continue

                    # gamma=0 必须复现同一主干的原 CL 结果。
                    if strength == 0:
                        reference = source_summary[
                            "test"
                        ][stage][task]["relative_l2"]

                        if not math.isclose(
                            metrics["relative_l2"],
                            reference,
                            rel_tol=1e-4,
                            abs_tol=1e-6,
                        ):
                            raise RuntimeError(
                                "gamma=0 未复现原 CL 测试结果，"
                                "停止比较。"
                            )

                    results[label][task] = {
                        k: v for k, v in metrics.items()
                        if k != "steps"
                    }
                    curves[(label, task)] = metrics["steps"]

                    save_csv(
                        folder / f"{label}_test_{task}.csv",
                        metrics["steps"],
                    )

                    part = parts[(task, "test")]
                    pred, truth = autoregressive_fields(
                        model,
                        part["states"],
                        part["forces"],
                        *norms,
                        grid,
                        100,
                        split["test"],
                        corrector=corrector(strength),
                    )

                    snapshots = [
                        (
                            step,
                            pred[0, step - 1].numpy(),
                            truth[0, step - 1].numpy(),
                        )
                        for step in (1, 10, 50, 100)
                    ]

                    save_field_comparison(
                        folder / f"{label}_test_{task}",
                        snapshots,
                    )

                    print(
                        label, task,
                        "test L2:", metrics["relative_l2"],
                        flush=True,
                    )

            if any(
                not torch.equal(
                    v.detach().cpu(), frozen[k]
                )
                for k, v in model.state_dict().items()
            ):
                raise RuntimeError("冻结主干发生变化。")

        fig, axes = plt.subplots(
            2, 2, figsize=(12, 8)
        )

        for row, task in enumerate(("A", "B")):
            for col, metric in enumerate((
                "relative_l2", "velocity_relative_l2"
            )):
                ax = axes[row, col]

                for (label, owner), rows in curves.items():
                    if owner == task:
                        ax.plot(
                            [r["time"] for r in rows],
                            [r[metric] for r in rows],
                            label=label,
                        )

                ax.set(
                    title=f"Task {task}: {metric}",
                    xlabel="Physical time",
                    ylabel="Relative L2",
                )
                ax.set_yscale(
                    "symlog", linthresh=1e-3
                )
                ax.grid(alpha=0.25)
                ax.legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(
            run_dir / "test_comparison.png", dpi=180
        )
        plt.close(fig)

        report = dict(
            status="completed",
            source=str(source),
            config=config,
            test=results,
            selection=selections,
            total_seconds=time.perf_counter() - started,
            limitations=[
                "单种子、每工况仅一条验证和测试轨迹。",
                "记忆训练额外使用64条旧转移和router计算，"
                "非等计算预算比较。",
                "A/B配对初态一致；检索时跨任务排除同pair。",
            ],
        )
        write_json(run_dir / "summary.json", report)

        lines = [
            "# CL + 可训练残差记忆试验",
            "",
            f"- 源实验：{source}",
            "- 主干冻结；每个主干单独重算残差。",
            "- 记忆：64条旧A转移 + 192条B转移。",
            "- router：B三步损失 + A单步损失。",
            "- epoch和gamma仅用A/B验证误差的均值选择。",
            "- 单种子小样本试验；额外计算和旧数据使用需另做消融。",
            "",
            "| 方法 | 工况 | 平均滚动 L2 | 速度 L2 |",
            "|---|---|---:|---:|",
        ]

        for label, values in results.items():
            for task, metrics in values.items():
                if metrics.get("status") == "nonfinite":
                    lines.append(
                        f"| {label} | {task}"
                        " | nonfinite | nonfinite |"
                    )
                else:
                    lines.append(
                        f"| {label} | {task}"
                        f" | {metrics['relative_l2']:.6f}"
                        f" | {metrics['velocity_relative_l2']:.6f} |"
                    )

        lines += [
            "",
            "![滚动误差](test_comparison.png)",
            "",
        ]

        for stage in ("finetune", "replay"):
            lines += [
                f"## {stage}，选择 gamma="
                f"{selections[stage]['gamma']}",
                "",
            ]

            for image in sorted(
                (run_dir / stage).glob("*.png")
            ):
                lines += [
                    f"![{image.stem}]({stage}/{image.name})",
                    "",
                ]

        (run_dir / "summary.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )

        print(
            "Summary：", run_dir / "summary.md",
            flush=True,
        )

    except BaseException as error:
        write_json(
            run_dir / "failure.json",
            dict(status="failed", error=repr(error)),
        )
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    if "--cl" in sys.argv:
        run_cl_memory()
    else:
        main()
