from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
import torch.nn.functional as F


# This file is expected at SWE_MemoryBank/memory_bank/residual_memory_bank.py.
ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "baseline_ffno"

if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from ffno_core.grid_2d import FNOFactorized2DBlock  # noqa: E402


def resolve_latest_file(
    explicit_path: str | Path | None,
    search_root: Path,
    filename: str,
) -> Path:
    """Resolve an explicit path or the most recently modified matching file."""
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        return path

    candidates = list(search_root.rglob(filename))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find {filename!r} below {search_root}. "
            "Set the path explicitly in CFG."
        )

    return max(candidates, key=lambda item: item.stat().st_mtime)


def choose_device(name: str) -> torch.device:
    """Select CPU/MPS/CUDA while keeping CPU as the safe default."""
    if name != "auto":
        return torch.device(name)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def build_model(config: dict, device: torch.device) -> torch.nn.Module:
    """Rebuild the FFNO architecture saved by train_ffno_swe.py."""
    model = FNOFactorized2DBlock(
        modes=int(config["modes"]),
        width=int(config["width"]),
        input_dim=3,
        n_layers=int(config["layers"]),
        factor=int(config["factor"]),
        n_ff_layers=2,
        share_weight=False,
        share_fork=False,
        ff_weight_norm=False,
        layer_norm=False,
        use_fork=False,
        dropout=0.0,
        in_dropout=0.0,
        mode="full",
    )
    return model.to(device)


def make_grid(
    nx: int,
    ny: int,
    device: torch.device,
) -> torch.Tensor:
    x = torch.linspace(0, 1, nx, device=device)
    y = torch.linspace(0, 1, ny, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    return torch.stack([xx, yy], dim=-1).unsqueeze(0)


def predict(
    model: torch.nn.Module,
    normalized_h: torch.Tensor,
    grid: torch.Tensor,
) -> torch.Tensor:
    coordinates = grid.expand(normalized_h.shape[0], -1, -1, -1)
    inputs = torch.cat([normalized_h, coordinates], dim=-1)
    return model(inputs)["forecast"]


def relative_l2(
    prediction: torch.Tensor,
    truth: torch.Tensor,
) -> torch.Tensor:
    prediction = prediction.to(torch.float64)
    truth = truth.to(torch.float64)
    difference = (prediction - truth).flatten(1)
    target = truth.flatten(1)
    numerator = torch.linalg.vector_norm(difference, dim=1)
    denominator = torch.linalg.vector_norm(target, dim=1).clamp_min(1e-12)
    return numerator / denominator


def load_split_ids(checkpoint_path: Path) -> dict[str, list[str]]:
    split_path = checkpoint_path.parent / "split_ids.json"
    if not split_path.is_file():
        raise FileNotFoundError(
            f"Missing {split_path}. The memory experiment must reuse the exact "
            "train/valid/test split from the FFNO baseline."
        )
    with split_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _downsample_block_average(array: np.ndarray, factor: int) -> np.ndarray:
    time_steps, nx, ny = array.shape
    if factor < 1 or nx % factor != 0 or ny % factor != 0:
        raise ValueError(
            f"downsample={factor} must divide spatial shape {(nx, ny)}."
        )
    return array.reshape(
        time_steps,
        nx // factor,
        factor,
        ny // factor,
        factor,
    ).mean(axis=(2, 4))


def load_trajectories_by_ids(
    data_path: Path,
    trajectory_ids: Iterable[str],
    downsample: int,
) -> tuple[torch.Tensor, np.ndarray]:
    """Load selected trajectories as [N,T,H,W,1] without reading all 1000."""
    trajectory_ids = list(trajectory_ids)
    trajectories: list[np.ndarray] = []
    reference_times: np.ndarray | None = None

    with h5py.File(data_path, "r") as handle:
        for index, trajectory_id in enumerate(trajectory_ids, start=1):
            if trajectory_id not in handle:
                raise KeyError(f"Trajectory {trajectory_id!r} is absent from {data_path}.")

            group = handle[trajectory_id]
            raw = np.asarray(group["data"][..., 0], dtype=np.float32)
            downsampled = _downsample_block_average(raw, downsample)

            if not np.isfinite(downsampled).all():
                raise ValueError(f"Trajectory {trajectory_id} contains NaN/Inf.")

            times = np.asarray(group["grid"]["t"][...], dtype=np.float64)
            if reference_times is None:
                reference_times = times
            elif not np.allclose(times, reference_times):
                raise ValueError("Selected trajectories use inconsistent time grids.")

            trajectories.append(downsampled)
            if index % 20 == 0 or index == len(trajectory_ids):
                print(
                    f"Loaded {index}/{len(trajectory_ids)} trajectories",
                    flush=True,
                )

    stacked = torch.from_numpy(np.stack(trajectories)).unsqueeze(-1)
    assert reference_times is not None
    return stacked, reference_times


def extract_query_features(
    normalized_h: torch.Tensor,
    raw_h: torch.Tensor,
    pool_size: int = 8,
    frequency_size: int = 4,
) -> torch.Tensor:
    """
    Build a compact, deterministic query/key descriptor.

    Inputs have shape [B,H,W,1]. Features contain:
    - pooled normalized water depth;
    - low-frequency Fourier magnitudes;
    - mean water depth, standard deviation, gradient strength, front radius.
    """
    if normalized_h.ndim != 4 or normalized_h.shape[-1] != 1:
        raise ValueError("normalized_h must have shape [B,H,W,1].")
    if raw_h.shape != normalized_h.shape:
        raise ValueError("raw_h and normalized_h must have identical shapes.")

    normalized_channels = normalized_h.permute(0, 3, 1, 2)
    pooled = F.adaptive_avg_pool2d(
        normalized_channels,
        output_size=(pool_size, pool_size),
    ).flatten(1)

    spectrum = torch.fft.rfft2(
        normalized_channels[:, 0],
        norm="ortho",
    ).abs()
    low_frequency = torch.log1p(
        spectrum[:, :frequency_size, :frequency_size]
    ).flatten(1)

    field = raw_h[..., 0]
    mean_depth = field.mean(dim=(1, 2), keepdim=False).unsqueeze(1)
    depth_std = field.std(dim=(1, 2), keepdim=False).unsqueeze(1)

    dx = field[:, 1:, :] - field[:, :-1, :]
    dy = field[:, :, 1:] - field[:, :, :-1]
    gradient_strength = (
        0.5
        * (
            dx.abs().mean(dim=(1, 2))
            + dy.abs().mean(dim=(1, 2))
        )
    ).unsqueeze(1)

    # Estimate radial wave-front location using gradient magnitude as weights.
    joint_gradient = torch.sqrt(
        dx[:, :, :-1].square() + dy[:, :-1, :].square() + 1e-12
    )
    gx, gy = joint_gradient.shape[1:]
    x_coordinates = torch.linspace(-1, 1, gx, device=field.device)
    y_coordinates = torch.linspace(-1, 1, gy, device=field.device)
    xx, yy = torch.meshgrid(x_coordinates, y_coordinates, indexing="ij")
    radius = torch.sqrt(xx.square() + yy.square()).unsqueeze(0)
    front_radius = (
        (joint_gradient * radius).sum(dim=(1, 2))
        / joint_gradient.sum(dim=(1, 2)).clamp_min(1e-12)
    ).unsqueeze(1)

    return torch.cat(
        [
            pooled,
            low_frequency,
            mean_depth,
            depth_std,
            gradient_strength,
            front_radius,
        ],
        dim=1,
    ).to(torch.float32)


def feature_group_slices(
    pool_size: int,
    frequency_size: int,
) -> list[tuple[int, int]]:
    """Slices of pooled image, low-frequency spectrum, and physics scalars."""
    pooled = int(pool_size) * int(pool_size)
    spectrum = int(frequency_size) * int(frequency_size)
    physics = 4
    return [
        (0, pooled),
        (pooled, pooled + spectrum),
        (pooled + spectrum, pooled + spectrum + physics),
    ]


def encode_memory_keys(
    raw_features: torch.Tensor,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    group_weights: torch.Tensor | None,
    group_slices: list[tuple[int, int]],
) -> torch.Tensor:
    """
    Standardize descriptors, then turn them into cosine-similarity keys.

    Without group weights, the whole 84-d vector is L2-normalized. The 64 pooled
    pixels then dominate retrieval. With weights, each group is normalized on its
    own and concatenated with scale sqrt(weight), so the key dot product is the
    weighted sum of the three group cosines.
    """
    standardized = (raw_features - feature_mean) / feature_std
    if group_weights is None:
        return F.normalize(standardized, p=2, dim=1)

    weights = group_weights.to(
        device=standardized.device,
        dtype=standardized.dtype,
    )
    weights = weights / weights.sum().clamp_min(1e-12)
    parts: list[torch.Tensor] = []
    for (start, end), weight in zip(group_slices, weights):
        group = F.normalize(standardized[:, start:end], p=2, dim=1)
        parts.append(group * torch.sqrt(weight.clamp_min(0.0)))
    return torch.cat(parts, dim=1)


class ResidualMemoryBank:
    """Read-only key-value bank: state descriptor -> normalized FFNO residual."""

    def __init__(
        self,
        keys: torch.Tensor,
        residuals: torch.Tensor,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
        metadata: dict,
        group_weights: torch.Tensor | None = None,
    ) -> None:
        if keys.ndim != 2:
            raise ValueError("keys must have shape [N,D].")
        if residuals.ndim != 4 or residuals.shape[-1] != 1:
            raise ValueError("residuals must have shape [N,H,W,1].")
        if len(keys) != len(residuals):
            raise ValueError("keys and residuals must contain the same entries.")

        self.keys = keys.to(torch.float32)
        self.residuals = residuals
        self.feature_mean = feature_mean.to(torch.float32)
        self.feature_std = feature_std.to(torch.float32)
        self.metadata = metadata
        self.pool_size = int(metadata["pool_size"])
        self.frequency_size = int(metadata["frequency_size"])
        if group_weights is None:
            self.group_weights = None
        else:
            weights = group_weights.to(torch.float32).flatten()
            if weights.numel() != 3 or not torch.isfinite(weights).all():
                raise ValueError(
                    "group_weights must contain three finite numbers: "
                    "pooled state, spectrum, physics."
                )
            if torch.any(weights < 0) or float(weights.sum()) <= 0:
                raise ValueError("group_weights must be non-negative and not all zero.")
            self.group_weights = weights

    @classmethod
    def from_raw_entries(
        cls,
        raw_features: torch.Tensor,
        residuals: torch.Tensor,
        metadata: dict,
        residual_dtype: torch.dtype = torch.float16,
        group_weights: list[float] | torch.Tensor | None = None,
    ) -> "ResidualMemoryBank":
        feature_mean = raw_features.mean(dim=0)
        feature_std = raw_features.std(dim=0).clamp_min(1e-6)
        weights = None if group_weights is None else torch.as_tensor(
            group_weights,
            dtype=torch.float32,
        )
        slices = feature_group_slices(
            int(metadata["pool_size"]),
            int(metadata["frequency_size"]),
        )
        keys = encode_memory_keys(
            raw_features,
            feature_mean,
            feature_std,
            weights,
            slices,
        )
        return cls(
            keys=keys.cpu(),
            residuals=residuals.to(residual_dtype).cpu(),
            feature_mean=feature_mean.cpu(),
            feature_std=feature_std.cpu(),
            metadata=metadata,
            group_weights=None if weights is None else weights.cpu(),
        )

    @property
    def size(self) -> int:
        return len(self.keys)

    @property
    def feature_dimension(self) -> int:
        return self.keys.shape[1]

    def to(self, device: torch.device) -> "ResidualMemoryBank":
        self.keys = self.keys.to(device)
        self.residuals = self.residuals.to(device)
        self.feature_mean = self.feature_mean.to(device)
        self.feature_std = self.feature_std.to(device)
        if self.group_weights is not None:
            self.group_weights = self.group_weights.to(device)
        return self

    def prepare_query(self, raw_features: torch.Tensor) -> torch.Tensor:
        return encode_memory_keys(
            raw_features.to(self.feature_mean.device),
            self.feature_mean,
            self.feature_std,
            self.group_weights,
            feature_group_slices(self.pool_size, self.frequency_size),
        )

    def retrieve(
        self,
        raw_query_features: torch.Tensor,
        top_k: int,
        temperature: float,
        minimum_similarity: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        top_k = min(max(int(top_k), 1), self.size)

        query = self.prepare_query(raw_query_features)
        similarities = query @ self.keys.T
        top_values, top_indices = torch.topk(similarities, k=top_k, dim=1)
        weights = torch.softmax(top_values / temperature, dim=1)

        selected_residuals = self.residuals[top_indices].to(query.dtype)
        correction = (
            weights[:, :, None, None, None] * selected_residuals
        ).sum(dim=1)

        maximum_similarity = top_values[:, 0]
        denominator = max(1.0 - minimum_similarity, 1e-6)
        gate = torch.clamp(
            (maximum_similarity - minimum_similarity) / denominator,
            min=0.0,
            max=1.0,
        )

        return correction, gate, maximum_similarity, top_indices

    def save(self, path: Path) -> None:
        payload = {
            "version": 1,
            "keys": self.keys.cpu(),
            "residuals": self.residuals.cpu(),
            "feature_mean": self.feature_mean.cpu(),
            "feature_std": self.feature_std.cpu(),
            "metadata": self.metadata,
        }
        if self.group_weights is not None:
            payload["group_weights"] = self.group_weights.cpu()
        torch.save(payload, path)

    @classmethod
    def load(
        cls,
        path: Path,
        device: torch.device,
    ) -> "ResidualMemoryBank":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        bank = cls(
            keys=payload["keys"],
            residuals=payload["residuals"],
            feature_mean=payload["feature_mean"],
            feature_std=payload["feature_std"],
            metadata=payload["metadata"],
            group_weights=payload.get("group_weights"),
        )
        return bank.to(device)
