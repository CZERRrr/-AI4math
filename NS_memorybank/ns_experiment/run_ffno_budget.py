"""在固定同工况划分上只训练 FFNO。

预算 A：10×20 次更新。预算 B：100×20 次更新。
两次加载同一份初始权重，并在训练开始时使用 seed=0。
epoch 表示固定次数的更新，不是把训练集完整走一遍。

归一化和训练样本只用训练轨迹的前 101 帧。
验证用 1、10、100 步，选择指标是 100 步联合 Relative L2。
测试在选择完成后评估 1、10、100、200 步。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

if os.environ.get("OMP_NUM_THREADS") == "0":
    os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ns_experiment.common import (  # noqa: E402
    RunLog,
    parameter_count,
    relative_l2,
    restore_rng,
    rng_state,
    seed_everything,
    write_json,
    write_markdown,
)
from ns_experiment.data import (  # noqa: E402
    load_cached_trajectories,
    population_mean_std,
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
    save_budget_curves,
    save_rollout_curves,
    save_truth_prediction,
)

CACHE_DIR = ROOT / "data" / "cache" / "nu0.01_512_down8"
BUDGETS = {"A": 10, "B": 100}
MODEL_CFG = {
    "width": 32,
    "modes": 12,
    "layers": 4,
    "factor": 4,
    "learning_rate": 1e-3,
    "batch_size": 8,
    "grad_clip": 1.0,
    "updates_per_epoch": 20,
    "seed": 0,
    "train_frames": 101,
    "validation_rollout_steps": [1, 10, 100],
    "test_rollout_steps": [1, 10, 100, 200],
    "selection_metric": "validation 100-step mean joint Relative L2",
}


def load_split_bundle():
    split_path = CACHE_DIR / "split_ids.json"
    manifest_path = CACHE_DIR / "manifest.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"还没有划分：{split_path}")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cache_files = [Path(item["path"]) for item in manifest["cache_files"]]
    records = load_cached_trajectories(cache_files)
    by_key = {record["identity"]["trajectory_key"]: record for record in records}

    def stack(group):
        chosen = [by_key[item["trajectory_key"]] for item in group]
        return {
            "identities": group,
            "states": torch.stack([item["states"] for item in chosen]),
            "forces": torch.stack([item["forces"] for item in chosen]),
            "times": torch.stack([item["times"] for item in chosen]),
        }

    parts = {name: stack(split[name]) for name in ("train", "valid", "test")}
    if parts["train"]["states"].shape[1] < 201:
        raise RuntimeError("缓存帧数少于 201，不能评估到 t=1.0。")
    return split, manifest, parts


def normalization_from_train(train_states, train_forces):
    frames = MODEL_CFG["train_frames"]
    state_mean, state_std = population_mean_std(train_states[:, :frames], (0, 1, 2, 3))
    force_mean, force_std = population_mean_std(train_forces, (0, 1, 2))
    return state_mean, state_std, force_mean, force_std


def prefix_mean(steps: list[dict], horizon: int, key: str) -> float:
    chunk = steps[:horizon]
    if len(chunk) != horizon:
        raise RuntimeError(f"滚动只有 {len(steps)} 步，不能汇总前 {horizon} 步。")
    return float(sum(float(row[key]) for row in chunk) / horizon)


def ensure_init(device: torch.device) -> Path:
    path = CACHE_DIR / "ffno_init_seed0.pt"
    if path.is_file():
        return path
    seed_everything(MODEL_CFG["seed"])
    model = build_ffno(MODEL_CFG)
    torch.save({"model": model.state_dict(), "config": MODEL_CFG, "seed": MODEL_CFG["seed"]}, path)
    print("已保存初始权重：", path, flush=True)
    return path


def save_checkpoint(path, model, optimizer, epoch, payload):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "completed_epoch": epoch,
            "rng": rng_state(),
            **payload,
        },
        path,
    )


def evaluate_horizons(model, raw_states, raw_forces, norms, grid, steps, identities):
    metrics = evaluate_autoregressive(
        model,
        raw_states,
        raw_forces,
        *norms,
        grid,
        steps,
        [item["index_in_file"] for item in identities],
    )
    # evaluate_autoregressive 的 trajectory id 只用于记忆排除。这里没有记忆，传文件内索引即可。
    return metrics


def forward_check() -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("前后向检查要求 CUDA。")
    device = torch.device("cuda")
    run_dir = ROOT / "results" / datetime.now().strftime("ns_ffno_forward_%Y%m%d_%H%M%S")
    run_dir.mkdir()
    split, manifest, parts = load_split_bundle()
    norms = normalization_from_train(parts["train"]["states"], parts["train"]["forces"])
    state_mean, state_std, force_mean, force_std = [item.to(device) for item in norms]
    init = ensure_init(device)
    seed_everything(MODEL_CFG["seed"])
    model = build_ffno(MODEL_CFG).to(device)
    model.load_state_dict(torch.load(init, map_location=device, weights_only=False)["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=MODEL_CFG["learning_rate"])
    frames = MODEL_CFG["train_frames"]
    trajectories = torch.randint(0, parts["train"]["states"].shape[0], (MODEL_CFG["batch_size"],))
    times = torch.randint(0, frames - 1, (MODEL_CFG["batch_size"],))
    state = (parts["train"]["states"][trajectories, times].to(device) - state_mean) / state_std
    target = (parts["train"]["states"][trajectories, times + 1].to(device) - state_mean) / state_std
    force = (parts["train"]["forces"][trajectories].to(device) - force_mean) / force_std
    grid = make_grid(*parts["train"]["states"].shape[2:4], device)
    prediction = forward_ffno(model, state, force, grid)
    loss = (prediction - target).square().mean()
    loss.backward()
    finite = all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
    if not finite or not torch.isfinite(loss):
        raise RuntimeError("小批量前后向出现非有限值。")
    total, trainable = parameter_count(model)
    report = {
        "status": "forward_backward_ok",
        "device": str(device),
        "loss": float(loss.detach()),
        "prediction_shape": list(prediction.shape),
        "parameters": total,
        "trainable": trainable,
        "trajectory_count": split["count"],
        "split": split["sizes"],
        "split_rule": split["rule"],
        "cache_bytes": manifest["cache_bytes"],
        "cache_dir": str(CACHE_DIR),
        "launch": "python ns_experiment/run_ffno_budget.py --budget A",
        "norm_frames": frames,
        "norm_uses_only_train_prefix": True,
    }
    write_json(run_dir / "summary.json", report)
    write_json(run_dir / "split_ids.json", split)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print("前后向检查：", run_dir, flush=True)
    return run_dir


def train_budget(budget: str, resume: Path | None) -> Path:
    if budget not in BUDGETS:
        raise ValueError("budget 必须是 A 或 B。")
    if not torch.cuda.is_available():
        raise RuntimeError("训练要求 CUDA。")
    device = torch.device("cuda")
    epochs = BUDGETS[budget]
    split, manifest, parts = load_split_bundle()
    norms = normalization_from_train(parts["train"]["states"], parts["train"]["forces"])
    init_path = ensure_init(device)
    if resume is None:
        run_dir = ROOT / "results" / datetime.now().strftime(f"ns_ffno_budget{budget}_%Y%m%d_%H%M%S")
        logger = RunLog(run_dir)
        start_epoch = 1
        best_l2 = float("inf")
        best_epoch = 0
        history = []
        seed_everything(MODEL_CFG["seed"])
        model = build_ffno(MODEL_CFG).to(device)
        model.load_state_dict(torch.load(init_path, map_location=device, weights_only=False)["model"])
        optimizer = torch.optim.Adam(model.parameters(), lr=MODEL_CFG["learning_rate"])
    else:
        run_dir = resume.resolve()
        logger = _AppendLog(run_dir)
        checkpoint = torch.load(run_dir / "last.pt", map_location=device, weights_only=False)
        if checkpoint["budget"] != budget:
            raise RuntimeError("恢复目录的预算和当前命令不一致。")
        model = build_ffno(MODEL_CFG).to(device)
        model.load_state_dict(checkpoint["model"])
        optimizer = torch.optim.Adam(model.parameters(), lr=MODEL_CFG["learning_rate"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        restore_rng(checkpoint["rng"])
        start_epoch = int(checkpoint["completed_epoch"]) + 1
        best_l2 = float(checkpoint["best_valid_l2"])
        best_epoch = int(checkpoint["best_epoch"])
        history = checkpoint["history"]
        print(f"从 epoch {start_epoch} 恢复：{run_dir}", flush=True)
    try:
        print("结果目录：", run_dir, flush=True)
        print("划分：", split["rule"], flush=True)
        config = {
            **MODEL_CFG,
            "budget": budget,
            "epochs": epochs,
            "init_checkpoint": str(init_path),
            "cache_dir": str(CACHE_DIR),
            "device": "cuda",
        }
        write_json(run_dir / "config.json", config)
        write_json(run_dir / "split_ids.json", split)
        write_json(run_dir / "data_manifest.json", manifest)
        state_mean, state_std, force_mean, force_std = norms
        state_mean_d = state_mean.to(device)
        state_std_d = state_std.to(device)
        force_mean_d = force_mean.to(device)
        force_std_d = force_std.to(device)
        grid = make_grid(*parts["train"]["states"].shape[2:4], device)
        train_states = parts["train"]["states"]
        train_forces = parts["train"]["forces"]
        frames = MODEL_CFG["train_frames"]
        total, _trainable = parameter_count(model)
        fields = [
            "epoch", "train_normalized_mse", "train_relative_l2",
            "valid_l2_1", "valid_l2_10", "valid_l2_100",
            "valid_velocity_l2_100", "valid_l2_vx_100", "valid_l2_vy_100",
            "valid_l2_particles_100", "seconds",
        ]
        history_path = run_dir / "history.csv"
        new_file = not history_path.is_file() or start_epoch == 1
        with history_path.open("a" if not new_file else "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if new_file:
                writer.writeheader()
            for epoch in range(start_epoch, epochs + 1):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                clock = datetime.now()
                model.train()
                loss_sum = 0.0
                l2_sum = 0.0
                seen = 0
                for _update in range(MODEL_CFG["updates_per_epoch"]):
                    batch = MODEL_CFG["batch_size"]
                    trajectories = torch.randint(0, train_states.shape[0], (batch,))
                    times = torch.randint(0, frames - 1, (batch,))
                    state = (train_states[trajectories, times].to(device) - state_mean_d) / state_std_d
                    target = (train_states[trajectories, times + 1].to(device) - state_mean_d) / state_std_d
                    force = (train_forces[trajectories].to(device) - force_mean_d) / force_std_d
                    optimizer.zero_grad(set_to_none=True)
                    prediction = forward_ffno(model, state, force, grid)
                    loss = (prediction - target).square().mean()
                    if not torch.isfinite(loss):
                        raise RuntimeError("训练损失出现 NaN/Inf。")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MODEL_CFG["grad_clip"])
                    optimizer.step()
                    loss_sum += float(loss.detach()) * batch
                    raw_prediction = prediction.detach() * state_std_d + state_mean_d
                    raw_target = target.detach() * state_std_d + state_mean_d
                    l2_sum += float(relative_l2(raw_prediction, raw_target).sum())
                    seen += batch
                valid = evaluate_horizons(
                    model, parts["valid"]["states"], parts["valid"]["forces"],
                    norms, grid, 100, split["valid"],
                )
                valid_100 = prefix_mean(valid["steps"], 100, "relative_l2")
                row = {
                    "epoch": epoch,
                    "train_normalized_mse": loss_sum / seen,
                    "train_relative_l2": l2_sum / seen,
                    "valid_l2_1": prefix_mean(valid["steps"], 1, "relative_l2"),
                    "valid_l2_10": prefix_mean(valid["steps"], 10, "relative_l2"),
                    "valid_l2_100": valid_100,
                    "valid_velocity_l2_100": prefix_mean(valid["steps"], 100, "velocity_relative_l2"),
                    "valid_l2_vx_100": prefix_mean(valid["steps"], 100, "relative_l2_velocity_x"),
                    "valid_l2_vy_100": prefix_mean(valid["steps"], 100, "relative_l2_velocity_y"),
                    "valid_l2_particles_100": prefix_mean(valid["steps"], 100, "relative_l2_particles"),
                    "seconds": (datetime.now() - clock).total_seconds(),
                }
                history.append(row)
                writer.writerow(row)
                handle.flush()
                if valid_100 < best_l2:
                    best_l2 = valid_100
                    best_epoch = epoch
                payload = {
                    "budget": budget,
                    "config": config,
                    "split_path": str(CACHE_DIR / "split_ids.json"),
                    "state_mean": state_mean,
                    "state_std": state_std,
                    "force_mean": force_mean,
                    "force_std": force_std,
                    "best_valid_l2": best_l2,
                    "best_epoch": best_epoch,
                    "history": history,
                    "init_checkpoint": str(init_path),
                }
                save_checkpoint(run_dir / "last.pt", model, optimizer, epoch, payload)
                if best_epoch == epoch:
                    save_checkpoint(run_dir / "best.pt", model, optimizer, epoch, payload)
                print(
                    f"Budget {budget} epoch {epoch:03d}/{epochs} | "
                    f"train L2 {row['train_relative_l2']:.4f} | "
                    f"valid 1/10/100 {row['valid_l2_1']:.4f}/"
                    f"{row['valid_l2_10']:.4f}/{row['valid_l2_100']:.4f}",
                    flush=True,
                )
        save_budget_curves(history, run_dir / "training_curves.png")
        checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        test = evaluate_horizons(
            model, parts["test"]["states"], parts["test"]["forces"],
            norms, grid, 200, split["test"],
        )
        clock = parts["test"]["times"][0]
        for row in test["steps"]:
            row["time"] = float(clock[row["step"]])
        with (run_dir / "test_rollout.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(test["steps"][0].keys()))
            writer.writeheader()
            writer.writerows(test["steps"])
        horizons = {}
        for horizon in MODEL_CFG["test_rollout_steps"]:
            horizons[str(horizon)] = {
                "joint_relative_l2": prefix_mean(test["steps"], horizon, "relative_l2"),
                "velocity_relative_l2": prefix_mean(test["steps"], horizon, "velocity_relative_l2"),
                "relative_l2_velocity_x": prefix_mean(test["steps"], horizon, "relative_l2_velocity_x"),
                "relative_l2_velocity_y": prefix_mean(test["steps"], horizon, "relative_l2_velocity_y"),
                "relative_l2_particles": prefix_mean(test["steps"], horizon, "relative_l2_particles"),
                "end_time": float(clock[horizon]),
            }
        late = test["steps"][100:]
        late_mean = float(sum(row["relative_l2"] for row in late) / len(late))
        prediction, truth = autoregressive_fields(
            model, parts["test"]["states"], parts["test"]["forces"],
            *norms, grid, 200, [item["index_in_file"] for item in split["test"]],
        )
        save_rollout_curves(
            test["steps"], run_dir / "test_rollout_curves.png",
            "Unseen trajectories", mark_step=100,
        )
        indices = [0, 9, 99, 199]
        times = [float(row["time"]) for row in test["steps"]]
        speed_truth = np.linalg.norm(truth[0, :, :, :, :2].numpy(), axis=-1)
        speed_pred = np.linalg.norm(prediction[0, :, :, :, :2].numpy(), axis=-1)
        save_truth_prediction(
            run_dir / "test_speed.png", speed_truth, speed_pred, times, indices,
            "Test speed. Step 200 reaches t=1.0, past the t=0.5 training window",
        )
        save_truth_prediction(
            run_dir / "test_particles.png",
            truth[0, :, :, :, 2].numpy(), prediction[0, :, :, :, 2].numpy(),
            times, indices,
            "Test particles. Step 200 reaches t=1.0, past the t=0.5 training window",
        )
        summary = {
            "status": "completed",
            "budget": budget,
            "updates": epochs * MODEL_CFG["updates_per_epoch"],
            "parameters": total,
            "best_epoch": best_epoch,
            "best_valid_100_relative_l2": best_l2,
            "test_horizons": horizons,
            "test_steps_101_to_200_joint_relative_l2": late_mean,
            "normalization": "train trajectories, frames 0:101 only",
            "init_checkpoint": str(init_path),
        }
        write_json(run_dir / "summary.json", summary)
        write_markdown(
            run_dir / "summary.md",
            f"NS FFNO 预算 {budget}",
            [
                ("状态", [
                    "completed",
                    f"更新次数：{summary['updates']}。epoch 是每 20 次更新记一次，不是完整遍历。",
                    f"参数量：{total}",
                    f"初始权重：{init_path}",
                    f"最佳 epoch：{best_epoch}，验证 100 步联合 Relative L2：{best_l2:.8f}",
                ]),
                ("数据", [
                    split["rule"],
                    "归一化、训练相邻帧和以后若建记忆，都只使用训练轨迹 t=0 到 0.5 的前 101 帧。",
                    f"缓存：{manifest['cache_bytes']} 字节，目录 {CACHE_DIR}",
                ]),
                ("100步", [
                    "未见轨迹上、从 t=0 滚到 t=0.5 的平均误差。这个时间范围和训练使用的帧一致。",
                    f"测试联合 Relative L2：{horizons['100']['joint_relative_l2']:.8f}",
                    f"速度：{horizons['100']['velocity_relative_l2']:.8f}",
                    f"vx/vy/particles：{horizons['100']['relative_l2_velocity_x']:.8f} / {horizons['100']['relative_l2_velocity_y']:.8f} / {horizons['100']['relative_l2_particles']:.8f}",
                ]),
                ("200步", [
                    "未见轨迹上滚到 t=1.0。其中 t>0.5 超出了训练使用的时间范围。",
                    f"前 200 步平均联合 Relative L2：{horizons['200']['joint_relative_l2']:.8f}",
                    f"第 101 到 200 步平均联合 Relative L2：{late_mean:.8f}",
                    f"速度：{horizons['200']['velocity_relative_l2']:.8f}",
                    f"vx/vy/particles：{horizons['200']['relative_l2_velocity_x']:.8f} / {horizons['200']['relative_l2_velocity_y']:.8f} / {horizons['200']['relative_l2_particles']:.8f}",
                ]),
                ("恢复", [
                    "每个完整 epoch 后写 last.pt，包含模型、优化器、已完成 epoch、随机状态和历史。",
                    "验证 100 步联合误差更小的 epoch 另写 best.pt。",
                    f"恢复命令：python ns_experiment/run_ffno_budget.py --budget {budget} --resume {run_dir}",
                    "中断在一个 epoch 内部时，只能恢复到上一个完整 epoch。",
                ]),
            ],
        )
        append_figure_notes(
            run_dir / "summary.md",
            ["training_curves.png", "test_rollout_curves.png", "test_speed.png", "test_particles.png"],
        )
        print("Summary：", run_dir / "summary.md", flush=True)
        return run_dir
    finally:
        if logger is not None:
            logger.close()


class _AppendLog:
    def __init__(self, run_dir: Path):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        self._handle = (run_dir / "run.log").open("a", encoding="utf-8")

        class _Tee:
            def __init__(self, *streams):
                self.streams = streams

            def write(self, data):
                for stream in self.streams:
                    stream.write(data)
                    stream.flush()
                return len(data)

            def flush(self):
                for stream in self.streams:
                    stream.flush()

        sys.stdout = _Tee(self._stdout, self._handle)
        sys.stderr = _Tee(self._stderr, self._handle)

    def close(self):
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        self._handle.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--budget", choices=["A", "B"])
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()
    if args.check:
        forward_check()
        return
    if args.budget is None:
        raise SystemExit("请指定 --check，或 --budget A / --budget B。")
    if args.resume is not None and not (args.resume / "last.pt").is_file():
        raise FileNotFoundError(args.resume / "last.pt")
    train_budget(args.budget, args.resume)


if __name__ == "__main__":
    main()
