from __future__ import annotations

import csv
import hashlib
import json
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from residual_memory_bank import (  # noqa: E402
    ROOT,
    ResidualMemoryBank,
    build_model,
    choose_device,
    extract_query_features,
    load_split_ids,
    load_trajectories_by_ids,
    make_grid,
    predict,
    relative_l2,
    resolve_latest_file,
    synchronize,
)


FINAL_BASELINE_DIR = ROOT / "results" / "ffno_swe_20261004_190400_532581"
EXPECTED_CHECKPOINT_EPOCH = 295
BANK_POINTER_PATH = FINAL_BASELINE_DIR / "final_memory_bank.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_final_bank_path() -> Path:
    if not BANK_POINTER_PATH.is_file():
        raise FileNotFoundError(
            "尚未生成最终模型的记忆库。请先运行 memory_bank/build_memory_bank.py。"
        )
    pointer = json.loads(BANK_POINTER_PATH.read_text(encoding="utf-8"))
    if int(pointer["checkpoint_epoch"]) != EXPECTED_CHECKPOINT_EPOCH:
        raise ValueError("记忆库指针不是 epoch 295 的最终实验，请重新建库。")
    expected_source = (FINAL_BASELINE_DIR / "best_model.pt").resolve()
    if Path(pointer["source_checkpoint_path"]).resolve() != expected_source:
        raise ValueError("记忆库来自其他 baseline 实验，请重新建库。")
    bank_path = Path(pointer["bank_path"]).expanduser().resolve()
    if not bank_path.is_file():
        raise FileNotFoundError(f"记忆库文件不存在：{bank_path}")
    if sha256_file(bank_path) != pointer["bank_sha256"]:
        raise ValueError("记忆库文件与建库记录不一致，请重新建库。")
    return bank_path


CFG = {
    # 自动读取本次最终 baseline 成功建库后保存的指针，不搜索旧库。
    "bank_path": None,
    # None means: use the checkpoint recorded inside memory_bank.pt.
    "checkpoint_path": None,
    # The AutoDL instance exposes an NVIDIA CUDA GPU.
    "device": "cuda",
    "batch_size": 32,
    # None 使用 baseline 划分中的全部轨迹：当前验证/测试各 100 条。
    "max_validation_trajectories": None,
    "max_test_trajectories": None,
    "top_k_candidates": [3, 5],
    "beta_candidates": [0.0, 0.25, 0.5, 0.75, 1.0],
    "temperature": 0.10,
    "minimum_similarity": 0.20,
    "seed": 0,
}


def limit_ids(ids: list[str], maximum: int | None) -> list[str]:
    if maximum is None:
        return ids
    return ids[: min(int(maximum), len(ids))]

@torch.no_grad()
def evaluate_rollout(
    model: torch.nn.Module,
    data: torch.Tensor,
    mean: float,
    std: float,
    grid: torch.Tensor,
    device: torch.device,
    bank: ResidualMemoryBank | None = None,
    beta: float = 0.0,
    top_k: int = 1,
    progress_label: str = "rollout",
    gate_mode: str = "original",
) -> dict:
    model.eval()
    horizon = data.shape[1] - 1
    l2_sum = np.zeros(horizon, dtype=np.float64)
    mse_sum = np.zeros(horizon, dtype=np.float64)
    similarity_sum = np.zeros(horizon, dtype=np.float64)
    gate_sum = np.zeros(horizon, dtype=np.float64)
    active_count = np.zeros(horizon, dtype=np.int64)
    total_examples = 0

    batches = data.split(int(CFG["batch_size"]))
    number_of_batches = len(batches)
    progress_interval = max(horizon // 4, 1)
    memory_enabled = bank is not None and beta > 0

    for batch_index, raw in enumerate(batches, start=1):
        raw = raw.to(device)
        current = (raw[:, 0] - mean) / std
        total_examples += len(raw)

        for step in range(horizon):
            base_prediction = predict(model, current, grid)

            if memory_enabled:
                current_raw = current * std + mean
                query_features = extract_query_features(
                    current,
                    current_raw,
                    pool_size=int(bank.metadata["pool_size"]),
                    frequency_size=int(bank.metadata["frequency_size"]),
                )
                correction, gate, maximum_similarity, _ = bank.retrieve(
                    query_features,
                    top_k=top_k,
                    temperature=float(CFG["temperature"]),
                    minimum_similarity=float(CFG["minimum_similarity"]),
                )

                if gate_mode == "off":
                    gate = torch.ones_like(gate)
                elif gate_mode != "original":
                    raise ValueError(f"Unknown gate_mode: {gate_mode}")

                current = (
                    base_prediction
                    + beta * gate[:, None, None, None] * correction
                )
                similarity_sum[step] += maximum_similarity.sum().item()
                gate_sum[step] += gate.sum().item()
                active_count[step] += int((gate > 0).sum().item())
            else:
                current = base_prediction

            if not torch.isfinite(current).all():
                raise RuntimeError(
                    f"Prediction diverged at rollout step {step + 1}."
                )

            prediction_raw = current.to(torch.float64) * std + mean
            truth = raw[:, step + 1].to(torch.float64)
            batch_l2 = relative_l2(prediction_raw, truth)
            batch_mse = (
                (prediction_raw - truth).square().flatten(1).mean(dim=1)
            )
            l2_sum[step] += batch_l2.sum().item()
            mse_sum[step] += batch_mse.sum().item()

            if (step + 1) % progress_interval == 0 or step + 1 == horizon:
                print(
                    f"{progress_label} | batch {batch_index}/{number_of_batches} "
                    f"| rollout step {step + 1}/{horizon}",
                    flush=True,
                )

    if total_examples == 0 or horizon < 1:
        raise ValueError("Evaluation requires trajectories with at least two frames.")

    l2_curve = l2_sum / total_examples
    mse_curve = mse_sum / total_examples
    similarity_curve = similarity_sum / total_examples
    gate_curve = gate_sum / total_examples
    active_rate_curve = active_count / total_examples

    total_predictions = total_examples * horizon
    active_predictions = int(active_count.sum())
    active_rate = active_predictions / total_predictions
    active_mean_gate = (
        float(gate_sum.sum()) / active_predictions
        if active_predictions > 0 else 0.0
    )

    if memory_enabled:
        print(
            f"{progress_label} | "
            f"memory active: {active_predictions}/{total_predictions} "
            f"({active_rate:.2%}) | "
            f"memory inactive: {1.0 - active_rate:.2%} | "
            f"mean gate when active: {active_mean_gate:.6f}",
            flush=True,
        )

        for start in range(0, horizon, 25):
            end = min(start + 25, horizon)
            segment_rate = float(active_rate_curve[start:end].mean())
            print(
                f"  steps {start + 1}-{end}: "
                f"memory active {segment_rate:.2%}",
                flush=True,
            )

    return {
        "l2": float(l2_curve.mean()),
        "mse": float(mse_curve.mean()),
        "final_l2": float(l2_curve[-1]),
        "l2_curve": l2_curve,
        "mse_curve": mse_curve,
        "similarity_curve": similarity_curve,
        "gate_curve": gate_curve,
        "mean_similarity": float(similarity_curve.mean()),
        "mean_gate": float(gate_curve.mean()),
        "active_rate_curve": active_rate_curve,
        "active_predictions": active_predictions,
        "total_predictions": total_predictions,
        "active_rate": float(active_rate),
        "active_mean_gate": float(active_mean_gate),
    }

@torch.no_grad()
def main() -> None:
    random.seed(CFG["seed"])
    np.random.seed(CFG["seed"])
    torch.manual_seed(CFG["seed"])

    device = choose_device(CFG["device"])
    print(f"Python executable: {sys.executable}", flush=True)
    print(f"Running script: {Path(__file__).resolve()}", flush=True)
    print(f"Device: {device} | PyTorch: {torch.__version__}", flush=True)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "This evaluation is configured for AutoDL CUDA. "
            "In VS Code use the SSH window and select the remote Python interpreter "
            "(/root/miniconda3/bin/python for your current instance). "
            "Do not run this GPU version with a local Mac interpreter."
        )
    print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)
    if CFG["bank_path"] is not None or CFG["checkpoint_path"] is not None:
        raise ValueError("本版本固定使用最终实验指针，请将这两个路径配置保持 None。")
    bank_path = resolve_final_bank_path()
    bank = ResidualMemoryBank.load(bank_path, device)

    if int(bank.metadata["checkpoint_epoch"]) != EXPECTED_CHECKPOINT_EPOCH:
        raise ValueError("读取到了旧模型的记忆库；请先用新建库脚本生成 epoch 295 的库。")
    if Path(bank.metadata["source_checkpoint_path"]).resolve() != (
        FINAL_BASELINE_DIR / "best_model.pt"
    ).resolve():
        raise ValueError("记忆库不属于本次最终 baseline 实验。")

    recorded_checkpoint = bank.metadata["checkpoint_path"]
    checkpoint_path = resolve_latest_file(
        CFG["checkpoint_path"] or recorded_checkpoint,
        ROOT / "results",
        "best_model.pt",
    )
    expected_hash = bank.metadata.get("snapshot_checkpoint_sha256")
    if expected_hash is None or sha256_file(checkpoint_path) != expected_hash:
        raise ValueError("模型快照与记忆库未配对，请先重新建库。")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model_config = checkpoint["config"]
    mean = float(checkpoint["mean"])
    std = float(checkpoint["std"])

    if int(checkpoint["epoch"]) != int(bank.metadata["checkpoint_epoch"]):
        raise ValueError(
            "The memory bank and checkpoint have different saved epochs. "
            "Rebuild the bank from this checkpoint."
        )
    if not np.isclose(mean, float(bank.metadata["mean"])) or not np.isclose(
        std, float(bank.metadata["std"])
    ):
        raise ValueError(
            "The memory bank and checkpoint use different normalization statistics. "
            "Rebuild the bank from this checkpoint."
        )

    model = build_model(model_config, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model_parameters = sum(parameter.numel() for parameter in model.parameters())

    split_ids = load_split_ids(checkpoint_path)
    valid_ids = split_ids["valid"]
    if set(split_ids["train"]) & set(valid_ids):
        raise ValueError("Train/validation IDs overlap")
    result_dir = ROOT / "results" / datetime.now().strftime("gate_ablation_valid_%Y%m%d_%H%M%S_%f")
    result_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "split": "valid", "validation_ids": valid_ids,
        "checkpoint_path": str(checkpoint_path), "checkpoint_epoch": int(checkpoint["epoch"]),
        "bank_path": str(bank_path), "bank_entries": bank.size,
        "top_k": 3, "beta": 1.0, "temperature": CFG["temperature"],
        "minimum_similarity": CFG["minimum_similarity"], "batch_size": CFG["batch_size"],
        "seed": CFG["seed"], "normalization_mean": mean, "normalization_std": std,
        "data_path": bank.metadata["data_path"], "downsample": bank.metadata["downsample"],
        "device": str(device), "gpu": torch.cuda.get_device_name(device),
        "python": sys.version, "torch": torch.__version__,
        "note": "Fixed k=3,beta=1 ablation; no hyperparameter search; validation only.",
    }
    (result_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Results directory: {result_dir}", flush=True)
    print("Loading validation data only...", flush=True)
    start_all = time.perf_counter()
    data, times = load_trajectories_by_ids(Path(bank.metadata["data_path"]), valid_ids, int(bank.metadata["downsample"]))
    grid = make_grid(*data.shape[2:4], device)
    config.update(trajectories=len(data), horizon=data.shape[1]-1, spatial_shape=list(data.shape[2:]), time_start=float(times[0]), time_end=float(times[-1]))
    results = {}
    curves = {}
    for name, use_bank, mode in [("baseline", False, "original"), ("memory_gated", True, "original"), ("memory_no_gate", True, "off")]:
        print(f"Starting {name}: k=3 beta={1.0 if use_bank else 0.0} gate={mode}", flush=True)
        synchronize(device)
        start_eval = time.perf_counter()
        try:
            result = evaluate_rollout(model, data, mean, std, grid, device,
                bank=bank if use_bank else None, beta=1.0 if use_bank else 0.0,
                top_k=3, progress_label=name, gate_mode=mode)
            synchronize(device)
            record = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
            record.update(status="completed", evaluation_seconds=time.perf_counter()-start_eval,
                          late_l2=float(result["l2_curve"][len(result["l2_curve"])//2:].mean()))
            if not all(np.isfinite(record[k]) for k in ["l2", "mse", "final_l2", "late_l2"]):
                raise RuntimeError("Non-finite evaluation metrics")
            curves[name] = result
            with (result_dir / f"{name}_steps.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["step", "time", "relative_l2", "mse", "mean_similarity", "mean_gate", "active_rate"])
                for i in range(len(result["l2_curve"])):
                    writer.writerow([i+1, float(times[i+1]), *[float(result[k][i]) for k in ["l2_curve", "mse_curve", "similarity_curve", "gate_curve", "active_rate_curve"]]])
            print(f"{name}: L2={record['l2']:.8f} final={record['final_l2']:.8f} late={record['late_l2']:.8f}", flush=True)
        except RuntimeError as exc:
            if "Prediction diverged" not in str(exc) and "Non-finite evaluation" not in str(exc):
                raise
            record = {"status": "diverged", "error": str(exc)}
            print(f"{name}: DIVERGED: {exc}", flush=True)
        results[name] = record
        (result_dir / "progress.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# Memory gate ablation — validation only", "", "Fixed FFNO epoch 295; same bank and validation trajectories. No training or test-set tuning.", "",
             "| Method | Mean rollout L2 | Final-step L2 | Second-half L2 | MSE | Active rate |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, r in results.items():
        if r["status"] == "completed":
            lines.append(f"| {name} | {r['l2']:.8f} | {r['final_l2']:.8f} | {r['late_l2']:.8f} | {r['mse']:.8f} | {r['active_rate']:.2%} |")
        else:
            lines.append(f"| {name} | DIVERGED | — | — | — | — |")
    if all(results[n]["status"] == "completed" for n in ["memory_gated", "memory_no_gate"]):
        change = (results["memory_gated"]["l2"]-results["memory_no_gate"]["l2"])/results["memory_gated"]["l2"]*100
        lines += ["", f"No-gate relative improvement over gated: {change:.2f}% (positive is better)."]
    lines += ["", "Active rate counts sample-steps with gate > 0; it does not measure helpful corrections.",
              "Timing is a single run without warm-up; do not interpret it as a rigorous speed benchmark.", "", "## Configuration", ""]
    lines += [f"- {k}: {v}" for k, v in config.items()]
    lines += ["", f"- Total seconds including data loading: {time.perf_counter()-start_all:.2f}"]
    (result_dir / "summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    (result_dir / "summary.json").write_text(json.dumps({"config": config, "results": results}, indent=2, allow_nan=False), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, r in curves.items():
        ax.plot(np.arange(1,len(r["l2_curve"])+1), r["l2_curve"], label=name)
    ax.set(xlabel="Forecast step", ylabel="Validation relative L2", title="Gate ablation (k=3, beta=1)")
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(result_dir / "comparison.png", dpi=180)
    plt.close(fig)
    print(f"\nCompleted. Summary: {result_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
