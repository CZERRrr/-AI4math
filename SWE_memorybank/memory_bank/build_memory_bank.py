from __future__ import annotations

import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from residual_memory_bank import (
    ROOT,
    ResidualMemoryBank,
    build_model,
    choose_device,
    extract_query_features,
    load_split_ids,
    load_trajectories_by_ids,
    make_grid,
    predict,
    resolve_latest_file,
    synchronize,
)


CFG = {
    # Set to None to automatically use the most recently modified best_model.pt.
    "checkpoint_path": None,
    "data_path": str(ROOT / "data" / "2D_rdb_NA_NA.h5"),
    "device": "cpu",
    "bank_trajectories": 100,
    "entries_per_trajectory": 10,
    # Choose "uniform" first; use "physics" for the main ablation.
    "selection_strategy": "uniform",
    # ground_truth stores residuals at true frames only.
    # pushforward rolls the frozen FFNO forward from each anchor and stores the
    # states the long rollout will actually query.
    "state_source": "pushforward",
    "pushforward_steps": 4,
    # Relative weights of pooled water depth, low-frequency spectrum, and the
    # four physics scalars. They are renormalized to sum to 1.
    "group_weights": [0.2, 0.3, 0.5],
    "build_batch_size": 32,
    "pool_size": 8,
    "frequency_size": 4,
    "residual_dtype": "float16",
    "seed": 0,
}


def normalized_scores(values: torch.Tensor) -> torch.Tensor:
    minimum = values.min()
    maximum = values.max()
    return (values - minimum) / (maximum - minimum).clamp_min(1e-12)


def select_transition_indices(
    trajectory: torch.Tensor,
    count: int,
    strategy: str,
) -> list[int]:
    """Select current-state indices t for transitions h_t -> h_(t+1)."""
    available = trajectory.shape[0] - 1
    count = min(max(int(count), 1), available)

    if strategy == "uniform":
        indices = np.linspace(0, available - 1, count, dtype=int)
        return sorted(np.unique(indices).tolist())

    if strategy != "physics":
        raise ValueError("selection_strategy must be 'uniform' or 'physics'.")

    field = trajectory[..., 0]
    temporal_change = torch.zeros(available)
    temporal_change[1:] = (
        (field[1:available] - field[: available - 1])
        .square()
        .flatten(1)
        .mean(dim=1)
        .sqrt()
    )

    selected_fields = field[:available]
    dx = selected_fields[:, 1:, :] - selected_fields[:, :-1, :]
    dy = selected_fields[:, :, 1:] - selected_fields[:, :, :-1]
    gradient = 0.5 * (
        dx.abs().mean(dim=(1, 2)) + dy.abs().mean(dim=(1, 2))
    )

    score = 0.5 * normalized_scores(temporal_change) + 0.5 * normalized_scores(
        gradient
    )
    indices = torch.topk(score, k=count).indices.tolist()
    return sorted(indices)


def method_lines() -> list[str]:
    if CFG["state_source"] == "pushforward":
        return [
            "Each anchor starts from a true training frame. The frozen FFNO is then",
            "rolled forward for several steps. At hop k the key describes the state",
            "the model itself produced, and the value is the residual that would",
            "correct that prediction toward the true next frame:",
            "",
            "$$",
            "k_{t,k} = \\Phi(c_{t,k}), \\qquad "
            "r_{t,k} = h_{t+k+1} - F_\\theta(c_{t,k}), "
            "\\qquad c_{t,0} = h_t, \\qquad c_{t,k+1} = F_\\theta(c_{t,k}).",
            "$$",
            "",
            "Pooled depth, low-frequency spectrum, and physics scalars are normalized",
            "as separate groups before cosine retrieval, so the wave-front features",
            "are not drowned by the pooled image. The bank uses training trajectories",
            "only and is read-only during validation and testing.",
        ]
    return [
        "Each entry stores a compact descriptor of a true training state as its key",
        "and the frozen FFNO's normalized one-step residual as its value:",
        "",
        "$$",
        "k_t = \\Phi(h_t), \\qquad r_t = h_{t+1} - F_\\theta(h_t).",
        "$$",
        "",
        "The bank uses training trajectories only. It is read-only during validation",
        "and testing, preventing test leakage and prediction-error contamination.",
        "Pooled depth, spectrum, and physics scalars can be weighted as separate",
        "key groups; omit group_weights to restore one-vector normalization.",
    ]


def anchor_prefix_length(num_frames: int, pushforward_steps: int) -> int:
    """Prefix length to pass into select_transition_indices.

    That helper emits current-state indices through length-2. The largest
    anchor t must still satisfy t + pushforward_steps <= num_frames - 1.
    """
    if pushforward_steps < 1:
        raise ValueError("pushforward_steps must be at least 1.")
    prefix_length = int(num_frames) - int(pushforward_steps) + 1
    if prefix_length < 2:
        raise ValueError(
            "The trajectory is too short for "
            f"pushforward_steps={pushforward_steps}."
        )
    return prefix_length


def write_build_summary(
    path: Path,
    source_checkpoint_path: Path,
    snapshot_checkpoint_path: Path,
    checkpoint: dict,
    bank: ResidualMemoryBank,
    result_dir: Path,
    seconds: float,
) -> None:
    title = (
        "# SWE Pushforward Residual Memory Bank — Build Summary"
        if CFG["state_source"] == "pushforward"
        else "# SWE Residual Memory Bank — Build Summary"
    )
    lines = [
        title,
        "",
        "## Source FFNO",
        "",
        f"- Original checkpoint: `{source_checkpoint_path}`",
        f"- Frozen checkpoint snapshot: `{snapshot_checkpoint_path}`",
        f"- Saved epoch: {checkpoint['epoch']}",
        f"- Saved validation rollout Relative L2: {checkpoint['valid_l2']:.8f}",
        f"- Normalization mean: {checkpoint['mean']:.10f}",
        f"- Normalization std: {checkpoint['std']:.10f}",
        "",
        "## Bank Configuration",
        "",
    ]
    for key, value in CFG.items():
        lines.append(f"- {key}: {value}")

    lines.extend(
        [
            "",
            "## Bank Result",
            "",
            f"- Number of memory entries: {bank.size}",
            f"- Query/key feature dimension: {bank.feature_dimension}",
            f"- Residual tensor dtype: {bank.residuals.dtype}",
            f"- Residual spatial shape: {list(bank.residuals.shape[1:])}",
            f"- Bank file size: {(result_dir / 'memory_bank.pt').stat().st_size / 1024**2:.2f} MiB",
            f"- Build time: {seconds:.2f} s",
            "",
            "## Method",
            "",
            *method_lines(),
            "",
            "## Artifacts",
            "",
            "- `memory_bank.pt`: serialized keys, residuals, feature normalization and metadata",
            "- `source_checkpoint.pt`: frozen FFNO checkpoint used to build the bank",
            "- `split_ids.json`: frozen baseline train/validation/test split",
            "- `build_config.json`: exact build configuration",
            "- `build_summary.md`: this report",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


@torch.no_grad()
def main() -> None:
    random.seed(CFG["seed"])
    np.random.seed(CFG["seed"])
    torch.manual_seed(CFG["seed"])

    device = choose_device(CFG["device"])
    data_path = Path(CFG["data_path"]).expanduser().resolve()
    checkpoint_path = resolve_latest_file(
        CFG["checkpoint_path"],
        ROOT / "results",
        "best_model.pt",
    )

    print(f"Using checkpoint: {checkpoint_path}", flush=True)
    print(f"Using device: {device}", flush=True)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model_config = checkpoint["config"]
    mean = float(checkpoint["mean"])
    std = float(checkpoint["std"])

    model = build_model(model_config, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    split_ids = load_split_ids(checkpoint_path)
    bank_ids = split_ids["train"][: int(CFG["bank_trajectories"])]
    trajectories, _ = load_trajectories_by_ids(
        data_path,
        bank_ids,
        int(model_config["downsample"]),
    )

    state_source = str(CFG["state_source"])
    if state_source not in {"ground_truth", "pushforward"}:
        raise ValueError("state_source must be 'ground_truth' or 'pushforward'.")
    pushforward_steps = (
        1 if state_source == "ground_truth" else int(CFG["pushforward_steps"])
    )

    anchor_states: list[torch.Tensor] = []
    anchor_trajectory_ids: list[str] = []
    anchor_times: list[int] = []
    trajectory_row = {trajectory_id: index for index, trajectory_id in enumerate(bank_ids)}

    for trajectory_id, trajectory in zip(bank_ids, trajectories):
        prefix_length = anchor_prefix_length(trajectory.shape[0], pushforward_steps)
        indices = select_transition_indices(
            trajectory[:prefix_length],
            int(CFG["entries_per_trajectory"]),
            str(CFG["selection_strategy"]),
        )
        for time_index in indices:
            anchor_states.append(trajectory[time_index])
            anchor_trajectory_ids.append(trajectory_id)
            anchor_times.append(int(time_index))

    current_raw = torch.stack(anchor_states)
    anchor_row = torch.tensor(
        [trajectory_row[trajectory_id] for trajectory_id in anchor_trajectory_ids],
        dtype=torch.long,
    )
    anchor_time = torch.tensor(anchor_times, dtype=torch.long)
    nx, ny = current_raw.shape[1:3]
    grid = make_grid(nx, ny, device)

    key_batches: list[torch.Tensor] = []
    residual_batches: list[torch.Tensor] = []
    entry_trajectory_ids: list[str] = []
    entry_time_indices: list[int] = []
    entry_horizons: list[int] = []
    build_start = time.perf_counter()
    batch_size = int(CFG["build_batch_size"])
    encoded = 0

    for hop in range(pushforward_steps):
        target_raw = trajectories[anchor_row, anchor_time + hop + 1]
        predicted_raw = torch.empty_like(current_raw)
        for start in range(0, len(current_raw), batch_size):
            end = min(start + batch_size, len(current_raw))
            raw_current = current_raw[start:end].to(device)
            raw_target = target_raw[start:end].to(device)
            normalized_current = (raw_current - mean) / std
            normalized_target = (raw_target - mean) / std

            base_prediction = predict(model, normalized_current, grid)
            residual = normalized_target - base_prediction
            features = extract_query_features(
                normalized_current,
                raw_current,
                pool_size=int(CFG["pool_size"]),
                frequency_size=int(CFG["frequency_size"]),
            )

            key_batches.append(features.cpu())
            residual_batches.append(residual.cpu())
            predicted_raw[start:end] = (base_prediction * std + mean).cpu()
            encoded += end - start

        entry_trajectory_ids.extend(anchor_trajectory_ids)
        entry_time_indices.extend((anchor_time + hop).tolist())
        entry_horizons.extend([hop] * len(anchor_trajectory_ids))
        print(
            f"Encoded hop {hop + 1}/{pushforward_steps} | entries so far {encoded}",
            flush=True,
        )
        if state_source == "pushforward":
            current_raw = predicted_raw

    synchronize(device)
    build_seconds = time.perf_counter() - build_start
    raw_features = torch.cat(key_batches, dim=0)
    residuals = torch.cat(residual_batches, dim=0)

    dtype = torch.float16 if CFG["residual_dtype"] == "float16" else torch.float32
    result_dir = (
        ROOT
        / "results"
        / datetime.now().strftime("memory_bank_%Y%m%d_%H%M%S_%f")
    )
    result_dir.mkdir(parents=True, exist_ok=False)
    snapshot_checkpoint_path = result_dir / "source_checkpoint.pt"
    torch.save(checkpoint, snapshot_checkpoint_path)
    with (result_dir / "split_ids.json").open("w", encoding="utf-8") as handle:
        json.dump(split_ids, handle, indent=2, ensure_ascii=False)

    metadata = {
        "checkpoint_path": str(snapshot_checkpoint_path),
        "source_checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_valid_l2": float(checkpoint["valid_l2"]),
        "data_path": str(data_path),
        "downsample": int(model_config["downsample"]),
        "mean": mean,
        "std": std,
        "pool_size": int(CFG["pool_size"]),
        "frequency_size": int(CFG["frequency_size"]),
        "selection_strategy": str(CFG["selection_strategy"]),
        "bank_trajectories_requested": int(CFG["bank_trajectories"]),
        "entries_per_trajectory": int(CFG["entries_per_trajectory"]),
        "build_batch_size": int(CFG["build_batch_size"]),
        "residual_dtype": str(CFG["residual_dtype"]),
        "seed": int(CFG["seed"]),
        "state_source": state_source,
        "pushforward_steps": pushforward_steps,
        "group_weights": (
            None
            if CFG["group_weights"] is None
            else [float(value) for value in CFG["group_weights"]]
        ),
        "bank_trajectory_ids": bank_ids,
        "entry_trajectory_ids": entry_trajectory_ids,
        "entry_time_indices": entry_time_indices,
        "entry_horizons": entry_horizons,
    }
    bank = ResidualMemoryBank.from_raw_entries(
        raw_features,
        residuals,
        metadata,
        residual_dtype=dtype,
        group_weights=None
        if CFG["group_weights"] is None
        else [float(value) for value in CFG["group_weights"]],
    )

    bank_path = result_dir / "memory_bank.pt"
    bank.save(bank_path)

    with (result_dir / "build_config.json").open("w", encoding="utf-8") as handle:
        json.dump(CFG, handle, indent=2, ensure_ascii=False)

    write_build_summary(
        result_dir / "build_summary.md",
        checkpoint_path,
        snapshot_checkpoint_path,
        checkpoint,
        bank,
        result_dir,
        build_seconds,
    )

    print("\nMemory bank build completed.")
    print(f"Entries: {bank.size}")
    print(f"Saved bank: {bank_path}")
    print(f"Summary: {result_dir / 'build_summary.md'}")


if __name__ == "__main__":
    main()
