from __future__ import annotations

import csv
import json
import platform
import random
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


CFG = {
    # None means: use the newest memory_bank.pt under results/.
    "bank_path": None,
    # None means: use the checkpoint recorded inside memory_bank.pt.
    "checkpoint_path": None,
    "device": "cpu",
    "batch_size": 8,
    # Small defaults for the first experiment. Set either to None for all trajectories.
    "max_validation_trajectories": 20,
    "max_test_trajectories": 20,
    "top_k_candidates": [3, 5],
    "beta_candidates": [0.0, 0.25, 0.5, 0.75, 1.0],
    "temperature": 0.10,
    # 0 lets a moderate cosine similarity contribute a proportional correction.
    # 0.2 reproduces the previous hard cutoff.
    "minimum_similarity_candidates": [0.0, 0.1, 0.2],
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
    minimum_similarity: float = 0.0,
    progress_label: str = "rollout",
) -> dict:
    model.eval()
    horizon = data.shape[1] - 1
    l2_sum = np.zeros(horizon, dtype=np.float64)
    mse_sum = np.zeros(horizon, dtype=np.float64)
    similarity_sum = np.zeros(horizon, dtype=np.float64)
    gate_sum = np.zeros(horizon, dtype=np.float64)
    total_examples = 0
    open_gate_count = 0
    total_gate_count = 0

    batches = data.split(int(CFG["batch_size"]))
    number_of_batches = len(batches)
    progress_interval = max(horizon // 4, 1)

    for batch_index, raw in enumerate(batches, start=1):
        raw = raw.to(device)
        current = (raw[:, 0] - mean) / std
        total_examples += len(raw)

        for step in range(horizon):
            base_prediction = predict(model, current, grid)

            if bank is not None and beta > 0:
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
                    minimum_similarity=float(minimum_similarity),
                )
                current = base_prediction + beta * gate[:, None, None, None] * correction
                similarity_sum[step] += maximum_similarity.sum().item()
                gate_sum[step] += gate.sum().item()
                open_gate_count += int((gate > 0).sum().item())
                total_gate_count += int(gate.numel())
            else:
                current = base_prediction

            if not torch.isfinite(current).all():
                raise RuntimeError(f"Prediction diverged at rollout step {step + 1}.")

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

    l2_curve = l2_sum / total_examples
    mse_curve = mse_sum / total_examples
    if bank is None or beta == 0:
        similarity_curve = np.zeros(horizon)
        gate_curve = np.zeros(horizon)
    else:
        similarity_curve = similarity_sum / total_examples
        gate_curve = gate_sum / total_examples

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
        "open_fraction": (
            float(open_gate_count) / float(total_gate_count)
            if total_gate_count
            else 0.0
        ),
    }


def save_step_csv(
    path: Path,
    times: np.ndarray,
    baseline: dict,
    memory: dict,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "forecast_step",
                "time",
                "baseline_relative_l2",
                "memory_relative_l2",
                "baseline_mse",
                "memory_mse",
                "memory_mean_similarity",
                "memory_mean_gate",
            ]
        )
        for index in range(len(baseline["l2_curve"])):
            writer.writerow(
                [
                    index + 1,
                    float(times[index + 1]),
                    float(baseline["l2_curve"][index]),
                    float(memory["l2_curve"][index]),
                    float(baseline["mse_curve"][index]),
                    float(memory["mse_curve"][index]),
                    float(memory["similarity_curve"][index]),
                    float(memory["gate_curve"][index]),
                ]
            )


def save_plot(path: Path, baseline: dict, memory: dict) -> None:
    steps = np.arange(1, len(baseline["l2_curve"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(steps, baseline["l2_curve"], label="FFNO baseline")
    axes[0].plot(steps, memory["l2_curve"], label="FFNO + memory")
    axes[0].set_xlabel("Forecast step")
    axes[0].set_ylabel("Relative L2")
    axes[0].set_title("Autoregressive rollout error")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(steps, memory["similarity_curve"], label="Top-1 similarity")
    axes[1].plot(steps, memory["gate_curve"], label="Memory gate")
    axes[1].set_xlabel("Forecast step")
    axes[1].set_ylabel("Value")
    axes[1].set_title("Retrieval confidence")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def markdown_table(rows: list[dict]) -> list[str]:
    lines = [
        "| top_k | beta | minimum similarity | Validation rollout Relative L2 |",
        "|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['top_k']} | {row['beta']:.2f} | "
            f"{row['minimum_similarity']:.2f} | {row['valid_l2']:.8f} |"
        )
    return lines


def write_summary(
    path: Path,
    checkpoint_path: Path,
    checkpoint: dict,
    bank_path: Path,
    bank: ResidualMemoryBank,
    validation_rows: list[dict],
    best_top_k: int,
    best_beta: float,
    best_minimum_similarity: float,
    baseline: dict,
    memory: dict,
    model_parameters: int,
    valid_count: int,
    test_count: int,
    spatial_shape: tuple[int, int],
    horizon: int,
    times: np.ndarray,
    device: torch.device,
    total_seconds: float,
) -> None:
    l2_improvement = 100.0 * (baseline["l2"] - memory["l2"]) / max(
        baseline["l2"], 1e-12
    )
    final_improvement = 100.0 * (
        baseline["final_l2"] - memory["final_l2"]
    ) / max(baseline["final_l2"], 1e-12)

    title = (
        "# FFNO + Pushforward Residual Memory Bank on PDEBench SWE"
        if bank.metadata.get("state_source") == "pushforward"
        else "# FFNO + Residual Memory Bank on PDEBench SWE"
    )
    lines = [
        title,
        "",
        "## Experiment Goal",
        "",
        "Evaluate whether a read-only residual memory bank reduces long-horizon "
        "autoregressive error relative to the same frozen FFNO checkpoint.",
        "",
        "## Baseline Checkpoint",
        "",
        f"- Checkpoint: `{checkpoint_path}`",
        f"- Saved epoch: {checkpoint['epoch']}",
        f"- Saved validation rollout Relative L2: {checkpoint['valid_l2']:.8f}",
        f"- Model parameters: {model_parameters:,}",
        f"- Normalization mean: {checkpoint['mean']:.10f}",
        f"- Normalization std: {checkpoint['std']:.10f}",
        "",
        "## Model Configuration",
        "",
    ]
    for key, value in checkpoint["config"].items():
        lines.append(f"- {key}: {value}")

    lines.extend(
        [
            "",
            "## Evaluation Data",
            "",
            f"- Data file: `{bank.metadata['data_path']}`",
            f"- Spatial resolution: {spatial_shape[0]} x {spatial_shape[1]}",
            f"- Rollout horizon: {horizon} steps",
            f"- Physical time range: {float(times[0]):.8g} to {float(times[-1]):.8g}",
            f"- Time-step size: {float(times[1] - times[0]):.8g}",
            f"- Validation trajectories used for tuning: {valid_count}",
            f"- Test trajectories used for final comparison: {test_count}",
            "- Split source: the exact `split_ids.json` produced by baseline training",
            f"- Evaluation batch size: {CFG['batch_size']}",
            f"- Random seed: {CFG['seed']}",
            f"- Device: {device}",
            f"- Python: {platform.python_version()}",
            f"- PyTorch: {torch.__version__}",
            f"- NumPy: {np.__version__}",
            f"- Platform: {platform.platform()}",
            "",
            "## Memory Bank",
            "",
            f"- Bank file: `{bank_path}`",
            f"- Number of entries: {bank.size}",
            f"- Key dimension: {bank.feature_dimension}",
            f"- Residual dtype: {bank.residuals.dtype}",
            f"- Selection strategy: {bank.metadata['selection_strategy']}",
            f"- Training trajectories represented: {len(bank.metadata['bank_trajectory_ids'])}",
            f"- Requested training trajectories: {bank.metadata.get('bank_trajectories_requested', 'not recorded')}",
            f"- Entries per trajectory: {bank.metadata.get('entries_per_trajectory', 'not recorded')}",
            f"- State source: {bank.metadata.get('state_source', 'ground_truth')}",
            f"- Pushforward steps: {bank.metadata.get('pushforward_steps', 1)}",
            f"- Key group weights (pool, spectrum, physics): {bank.metadata.get('group_weights', 'uniform vector')}",
            f"- Bank build batch size: {bank.metadata.get('build_batch_size', 'not recorded')}",
            f"- Bank build seed: {bank.metadata.get('seed', 'not recorded')}",
            f"- Bank file size: {bank_path.stat().st_size / 1024**2:.2f} MiB",
            "- Value: normalized one-step FFNO residual",
            "- Key: pooled state, low-frequency spectrum, mean, standard deviation, "
            "gradient strength and wave-front radius",
            "- Write policy: offline training states only; no validation/test writes",
            "",
            "## Retrieval Configuration",
            "",
            f"- top_k candidates: {CFG['top_k_candidates']}",
            f"- beta candidates: {CFG['beta_candidates']}",
            f"- minimum similarity candidates: {CFG['minimum_similarity_candidates']}",
            f"- softmax temperature: {CFG['temperature']}",
            f"- selected top_k: {best_top_k}",
            f"- selected beta: {best_beta}",
            f"- selected minimum similarity: {best_minimum_similarity}",
            "",
            "## Validation Hyperparameter Search",
            "",
            *markdown_table(validation_rows),
            "",
            "## Final Test Results",
            "",
            "| Method | Mean rollout Relative L2 | Final-step Relative L2 | Mean rollout MSE |",
            "|---|---:|---:|---:|",
            f"| FFNO baseline | {baseline['l2']:.8f} | {baseline['final_l2']:.8f} | {baseline['mse']:.8f} |",
            f"| FFNO + memory | {memory['l2']:.8f} | {memory['final_l2']:.8f} | {memory['mse']:.8f} |",
            "",
            f"- Mean rollout Relative L2 improvement: **{l2_improvement:.2f}%**",
            f"- Final-step Relative L2 improvement: **{final_improvement:.2f}%**",
            f"- Mean retrieved top-1 similarity: {memory['mean_similarity']:.6f}",
            f"- Mean memory gate: {memory['mean_gate']:.6f}",
            f"- Fraction of test steps with gate > 0: {memory['open_fraction']:.6f}",
            f"- Total evaluation time: {total_seconds:.2f} s",
            "",
            "Positive improvement means the memory model has lower error. Negative "
            "improvement means the selected memory correction degraded the baseline.",
            "",
            "## Experimental Controls",
            "",
            "- The FFNO weights remain frozen for both methods.",
            "- Baseline and memory methods use the same checkpoint, test trajectories and horizon.",
            "- Hyperparameters are selected on validation trajectories only.",
            "- The memory bank contains training trajectories only.",
            "- The bank is checkpoint-specific because its values are FFNO residuals.",
            "",
            "## Artifacts",
            "",
            "- `summary.md`: full experiment report",
            "- `summary.json`: machine-readable results and configuration",
            "- `validation_search.csv`: validation hyperparameter search",
            "- `rollout_comparison.csv`: step-wise test errors and retrieval confidence",
            "- `rollout_comparison.png`: error and confidence curves",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


@torch.no_grad()
def main() -> None:
    random.seed(CFG["seed"])
    np.random.seed(CFG["seed"])
    torch.manual_seed(CFG["seed"])

    device = choose_device(CFG["device"])
    bank_path = resolve_latest_file(
        CFG["bank_path"],
        ROOT / "results",
        "memory_bank.pt",
    )
    bank = ResidualMemoryBank.load(bank_path, device)

    recorded_checkpoint = bank.metadata["checkpoint_path"]
    checkpoint_path = resolve_latest_file(
        CFG["checkpoint_path"] or recorded_checkpoint,
        ROOT / "results",
        "best_model.pt",
    )
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
    valid_ids = limit_ids(
        split_ids["valid"],
        CFG["max_validation_trajectories"],
    )
    test_ids = limit_ids(
        split_ids["test"],
        CFG["max_test_trajectories"],
    )
    data_path = Path(bank.metadata["data_path"])
    downsample = int(bank.metadata["downsample"])
    validation, validation_times = load_trajectories_by_ids(
        data_path,
        valid_ids,
        downsample,
    )
    test, test_times = load_trajectories_by_ids(
        data_path,
        test_ids,
        downsample,
    )
    if not np.allclose(validation_times, test_times):
        raise ValueError("Validation and test time grids differ.")

    nx, ny = test.shape[2:4]
    grid = make_grid(nx, ny, device)
    result_dir = (
        ROOT
        / "results"
        / datetime.now().strftime("memory_eval_%Y%m%d_%H%M%S_%f")
    )
    result_dir.mkdir(parents=True, exist_ok=False)

    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Using memory bank: {bank_path}")
    print(f"Device: {device}")
    print(
        f"Validation trajectories: {len(validation)} | Test trajectories: {len(test)}",
        flush=True,
    )

    evaluation_start = time.perf_counter()
    validation_rows: list[dict] = []
    best_valid_l2 = float("inf")
    best_top_k = 1
    best_beta = 0.0
    best_minimum_similarity = float(CFG["minimum_similarity_candidates"][0])

    for top_k in CFG["top_k_candidates"]:
        for beta in CFG["beta_candidates"]:
            # beta = 0 is the frozen FFNO. Retrieval settings do not affect it.
            if float(beta) == 0.0 and int(top_k) != int(CFG["top_k_candidates"][0]):
                continue
            similarity_candidates = (
                [float(CFG["minimum_similarity_candidates"][0])]
                if float(beta) == 0.0
                else CFG["minimum_similarity_candidates"]
            )
            for minimum_similarity in similarity_candidates:
                result = evaluate_rollout(
                    model,
                    validation,
                    mean,
                    std,
                    grid,
                    device,
                    bank=bank,
                    beta=float(beta),
                    top_k=int(top_k),
                    minimum_similarity=float(minimum_similarity),
                    progress_label=(
                        f"validation k={top_k} beta={float(beta):.2f} "
                        f"min_sim={float(minimum_similarity):.2f}"
                    ),
                )
                row = {
                    "top_k": int(top_k),
                    "beta": float(beta),
                    "minimum_similarity": float(minimum_similarity),
                    "valid_l2": result["l2"],
                    "open_fraction": result["open_fraction"],
                }
                validation_rows.append(row)
                print(
                    f"Validation | top_k={top_k} beta={float(beta):.2f} "
                    f"min_sim={float(minimum_similarity):.2f} "
                    f"rollout L2={result['l2']:.8f} "
                    f"open={result['open_fraction']:.3f}",
                    flush=True,
                )
                if result["l2"] < best_valid_l2:
                    best_valid_l2 = result["l2"]
                    best_top_k = int(top_k)
                    best_beta = float(beta)
                    best_minimum_similarity = float(minimum_similarity)

    print("Evaluating baseline on test split...", flush=True)
    baseline = evaluate_rollout(
        model,
        test,
        mean,
        std,
        grid,
        device,
        progress_label="test baseline",
    )
    print(
        "Evaluating memory model with "
        f"top_k={best_top_k}, beta={best_beta}, "
        f"minimum_similarity={best_minimum_similarity}...",
        flush=True,
    )
    memory = evaluate_rollout(
        model,
        test,
        mean,
        std,
        grid,
        device,
        bank=bank,
        beta=best_beta,
        top_k=best_top_k,
        minimum_similarity=best_minimum_similarity,
        progress_label="test memory",
    )
    synchronize(device)
    total_seconds = time.perf_counter() - evaluation_start

    with (result_dir / "validation_search.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "top_k",
                "beta",
                "minimum_similarity",
                "valid_l2",
                "open_fraction",
            ],
        )
        writer.writeheader()
        writer.writerows(validation_rows)

    save_step_csv(
        result_dir / "rollout_comparison.csv",
        test_times,
        baseline,
        memory,
    )
    save_plot(result_dir / "rollout_comparison.png", baseline, memory)

    summary_json = {
        "config": CFG,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "bank_path": str(bank_path),
        "bank_entries": bank.size,
        "validation_trajectories": len(validation),
        "test_trajectories": len(test),
        "selected_top_k": best_top_k,
        "selected_beta": best_beta,
        "selected_minimum_similarity": best_minimum_similarity,
        "best_validation_l2": best_valid_l2,
        "baseline": {
            "l2": baseline["l2"],
            "mse": baseline["mse"],
            "final_l2": baseline["final_l2"],
        },
        "memory": {
            "l2": memory["l2"],
            "mse": memory["mse"],
            "final_l2": memory["final_l2"],
            "mean_similarity": memory["mean_similarity"],
            "mean_gate": memory["mean_gate"],
            "open_fraction": memory["open_fraction"],
        },
        "total_seconds": total_seconds,
    }
    with (result_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_json, handle, indent=2, ensure_ascii=False)

    write_summary(
        result_dir / "summary.md",
        checkpoint_path,
        checkpoint,
        bank_path,
        bank,
        validation_rows,
        best_top_k,
        best_beta,
        best_minimum_similarity,
        baseline,
        memory,
        model_parameters,
        len(validation),
        len(test),
        (nx, ny),
        test.shape[1] - 1,
        test_times,
        device,
        total_seconds,
    )

    print("\nMemory experiment completed.")
    print(f"Baseline rollout L2: {baseline['l2']:.8f}")
    print(f"Memory rollout L2:   {memory['l2']:.8f}")
    print(f"Summary: {result_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
