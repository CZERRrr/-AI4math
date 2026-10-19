"""条件 FFNO、固定特征，以及固定或可训练的残差修正。

频谱块仍使用原来的 FNOFactorized2DBlock。非 fork 路径保持
forecast = out(b)。这里只把输出通道从 1 改成状态通道数。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "baseline_ffno"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from ffno_core.grid_2d import FNOFactorized2DBlock  # noqa: E402


def build_ffno(config: dict) -> FNOFactorized2DBlock:
    model = FNOFactorized2DBlock(modes=int(config['modes']), width=int(config['width']), input_dim=7, output_dim=3, n_layers=int(config['layers']), factor=int(config['factor']), n_ff_layers=2, share_weight=False, share_fork=False, ff_weight_norm=False, layer_norm=False, use_fork=False, dropout=0.0, in_dropout=0.0, mode='full')
    mode = config.get('prediction_mode', 'direct')
    if mode not in ('direct', 'increment'):
        raise ValueError(f'未知预测模式：{mode}')
    model.prediction_mode = mode
    return model


def make_grid(height: int, width: int, device: torch.device) -> torch.Tensor:
    x = torch.linspace(0, 1, height, device=device)
    y = torch.linspace(0, 1, width, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    return torch.stack([xx, yy], dim=-1).unsqueeze(0)


def forward_ffno(model: nn.Module, state: torch.Tensor, force: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """state、force 都是已标准化场。grid 为 [1,H,W,2] 或 [B,H,W,2]。"""
    if force.shape[0] != state.shape[0]:
        force = force.expand(state.shape[0], -1, -1, -1)
    if grid.shape[0] != state.shape[0]:
        grid = grid.expand(state.shape[0], -1, -1, -1)
    prediction = model(torch.cat([state, force, grid], dim=-1))['forecast']
    mode = getattr(model, 'prediction_mode', 'direct')
    if mode == 'increment':
        return state + prediction
    if mode == 'direct':
        return prediction
    raise ValueError(f'未知预测模式：{mode}')


def state_force_features(state: torch.Tensor, force: torch.Tensor) -> torch.Tensor:
    """固定 query/key。state [B,H,W,3]，force [B,H,W,2]。"""
    state_nchw = state.permute(0, 3, 1, 2)
    force_nchw = force.permute(0, 3, 1, 2)
    pooled = F.adaptive_avg_pool2d(state_nchw, 4).flatten(1)
    spectrum = torch.fft.rfft2(state_nchw, norm="ortho")[..., :4, :4].abs().flatten(1)
    mean = state.mean(dim=(1, 2))
    std = state.std(dim=(1, 2), unbiased=False)
    grad_x = (state[:, 1:] - state[:, :-1]).abs().mean(dim=(1, 2))
    grad_y = (state[:, :, 1:] - state[:, :, :-1]).abs().mean(dim=(1, 2))
    force_pooled = F.adaptive_avg_pool2d(force_nchw, 4).flatten(1)
    force_mean = force.mean(dim=(1, 2))
    force_std = force.std(dim=(1, 2), unbiased=False)
    return torch.cat(
        [pooled, spectrum, mean, std, grad_x, grad_y, force_pooled, force_mean, force_std],
        dim=-1,
    )


def residual_stats(residual: torch.Tensor) -> torch.Tensor:
    mean = residual.mean(dim=(-3, -2))
    std = residual.std(dim=(-3, -2), unbiased=False)
    magnitude = residual.flatten(-3).norm(dim=-1, keepdim=True)
    return torch.cat([mean, std, magnitude], dim=-1)


class CandidateScorer(nn.Module):
    """给检索残差和零残差候选打分。离散 Top-K 本身不参加反传。"""

    def __init__(self, feature_dim: int, stat_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.stat_dim = stat_dim
        self.net = nn.Sequential(
            nn.Linear(feature_dim * 3 + stat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def score(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        stats: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(torch.cat([query, key, query - key, stats], dim=-1)).squeeze(-1)


class MemoryBank:
    def __init__(
        self,
        keys: torch.Tensor,
        residuals: torch.Tensor,
        trajectory_ids: torch.Tensor,
        time_indices: torch.Tensor,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
    ):
        self.keys = keys
        self.residuals = residuals
        self.trajectory_ids = trajectory_ids.long()
        self.time_indices = time_indices.long()
        self.feature_mean = feature_mean
        self.feature_std = feature_std

    @property
    def size(self) -> int:
        return int(self.keys.shape[0])

    def save(self, path: Path) -> None:
        torch.save(
            {
                "keys": self.keys.cpu(),
                "residuals": self.residuals.cpu().to(torch.float16),
                "trajectory_ids": self.trajectory_ids.cpu(),
                "time_indices": self.time_indices.cpu(),
                "feature_mean": self.feature_mean.cpu(),
                "feature_std": self.feature_std.cpu(),
                "feature_version": "pool4_fft4_stats_grad_force_v1",
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, device: torch.device) -> "MemoryBank":
        payload = torch.load(path, map_location=device, weights_only=False)
        return cls(
            keys=payload["keys"].to(device=device, dtype=torch.float32),
            residuals=payload["residuals"].to(device=device, dtype=torch.float32),
            trajectory_ids=payload["trajectory_ids"].to(device),
            time_indices=payload["time_indices"].to(device),
            feature_mean=payload["feature_mean"].to(device=device, dtype=torch.float32),
            feature_std=payload["feature_std"].to(device=device, dtype=torch.float32),
        )


def standardize_features(
    features: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    return (features - mean) / std


@torch.no_grad()
def build_memory_bank(
    model: nn.Module,
    states: torch.Tensor,
    forces: torch.Tensor,
    grid: torch.Tensor,
    trajectory_ids: list[int],
    entry_budget: int,
    device: torch.device,
    seed: int,
) -> MemoryBank:
    """states、forces 已按训练集标准化。残差来自当前冻结主干。"""
    model.eval()
    count, steps = states.shape[:2]
    transitions = count * (steps - 1)
    budget = min(int(entry_budget), transitions)
    if budget < 1:
        raise RuntimeError("没有可用的训练转移来构建记忆库。")
    chosen = choose_entries(transitions, budget, seed)
    raw_features = []
    residuals = []
    owners = []
    times = []
    for flat in chosen:
        trajectory, time_index = divmod(int(flat), steps - 1)
        state = states[trajectory, time_index].unsqueeze(0).to(device)
        target = states[trajectory, time_index + 1].unsqueeze(0).to(device)
        force = forces[trajectory].unsqueeze(0).to(device)
        prediction = forward_ffno(model, state, force, grid)
        raw_features.append(state_force_features(state, force).squeeze(0).cpu())
        residuals.append((target - prediction).squeeze(0).cpu())
        owners.append(trajectory_ids[trajectory])
        times.append(time_index)
    stacked = torch.stack(raw_features)
    feature_mean = stacked.mean(dim=0)
    feature_std = stacked.std(dim=0, unbiased=False).clamp_min(1e-6)
    return MemoryBank(
        keys=standardize_features(stacked, feature_mean, feature_std).to(device),
        residuals=torch.stack(residuals).to(device),
        trajectory_ids=torch.tensor(owners, dtype=torch.long, device=device),
        time_indices=torch.tensor(times, dtype=torch.long, device=device),
        feature_mean=feature_mean.to(device),
        feature_std=feature_std.to(device),
    )


def choose_entries(population: int, count: int, seed: int) -> list[int]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randperm(population, generator=generator)[:count].tolist()


def eligible_indices(bank: MemoryBank, trajectory_id: int) -> torch.Tensor:
    return torch.where(bank.trajectory_ids != int(trajectory_id))[0]


def retrieve(
    bank: MemoryBank,
    query_standard: torch.Tensor,
    trajectory_id: int,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回本条轨迹之外的 Top-K。索引不可微，返回值不带检索图。"""
    allowed = eligible_indices(bank, trajectory_id)
    if allowed.numel() == 0:
        empty = allowed
        return empty, query_standard.new_zeros((0,))
    k_eff = min(int(k), int(allowed.numel()))
    query = F.normalize(query_standard, dim=0)
    keys = F.normalize(bank.keys.index_select(0, allowed), dim=-1)
    similarity = keys @ query.detach()
    _values, local = torch.topk(similarity, k=k_eff)
    chosen = allowed.index_select(0, local)
    return chosen, similarity.index_select(0, local).detach()


def fixed_delta(
    bank: MemoryBank,
    state: torch.Tensor,
    force: torch.Tensor,
    trajectory_ids: torch.Tensor,
    k: int,
    temperature: float,
    gamma: float,
) -> torch.Tensor:
    features = standardize_features(
        state_force_features(state, force),
        bank.feature_mean,
        bank.feature_std,
    )
    deltas = []
    for row in range(state.shape[0]):
        if gamma == 0:
            deltas.append(torch.zeros_like(state[row]))
            continue
        chosen, similarity = retrieve(bank, features[row], int(trajectory_ids[row]), k)
        if chosen.numel() == 0:
            deltas.append(torch.zeros_like(state[row]))
            continue
        weights = torch.softmax(similarity / temperature, dim=0)
        residual = bank.residuals.index_select(0, chosen)
        deltas.append(torch.einsum("k,khwc->hwc", weights, residual) * gamma)
    return torch.stack(deltas)


def learned_delta(
    scorer: CandidateScorer,
    bank: MemoryBank,
    state: torch.Tensor,
    force: torch.Tensor,
    trajectory_ids: torch.Tensor,
    k: int,
    gamma: float,
) -> torch.Tensor:
    features = standardize_features(
        state_force_features(state, force),
        bank.feature_mean,
        bank.feature_std,
    )
    zero_stats = residual_stats(state.new_zeros(state.shape[1:]))
    deltas = []
    for row in range(state.shape[0]):
        if gamma == 0:
            deltas.append(torch.zeros_like(state[row]))
            continue
        chosen, _similarity = retrieve(bank, features[row], int(trajectory_ids[row]), k)
        query = features[row]
        scores = []
        residuals = []
        for index in chosen.tolist():
            key = bank.keys[index]
            residual = bank.residuals[index]
            scores.append(scorer.score(query, key, residual_stats(residual)))
            residuals.append(residual)
        scores.append(scorer.score(query, torch.zeros_like(query), zero_stats))
        weights = torch.softmax(torch.stack(scores), dim=0)
        if residuals:
            mixed = torch.einsum(
                "k,khwc->hwc",
                weights[:-1],
                torch.stack(residuals),
            )
        else:
            mixed = torch.zeros_like(state[row])
        deltas.append(mixed * gamma)
    return torch.stack(deltas)


def rollout(
    model: nn.Module,
    initial_state: torch.Tensor,
    force: torch.Tensor,
    grid: torch.Tensor,
    steps: int,
    corrector=None,
) -> torch.Tensor:
    """返回标准化空间中的逐步预测 [B,steps,H,W,C]。"""
    current = initial_state
    outputs = []
    for _ in range(steps):
        prediction = forward_ffno(model, current, force, grid)
        if corrector is not None:
            prediction = prediction + corrector(current)
        current = prediction
        outputs.append(current)
    return torch.stack(outputs, dim=1)


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def assert_later_ffno_sees_correction(
    model: nn.Module,
    state: torch.Tensor,
    force: torch.Tensor,
    grid: torch.Tensor,
    target: torch.Tensor,
) -> None:
    """确认下一步 FFNO 对修正后的输入保留梯度。"""
    delta = torch.zeros_like(state, requires_grad=True)
    corrected = state + delta
    prediction = forward_ffno(model, corrected, force, grid)
    loss = (prediction - target).square().mean()
    gradient = torch.autograd.grad(loss, delta, allow_unused=True)[0]
    if gradient is None or not torch.isfinite(gradient).all() or float(gradient.abs().sum()) == 0:
        raise RuntimeError("修正量没有传入下一步 FFNO。不能在逐步调用外包 no_grad。")


@torch.no_grad()
def evaluate_autoregressive(
    model: nn.Module,
    raw_states: torch.Tensor,
    raw_forces: torch.Tensor,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    force_mean: torch.Tensor,
    force_std: torch.Tensor,
    grid: torch.Tensor,
    steps: int,
    trajectory_ids: list[int],
    corrector=None,
) -> dict:
    """在原始尺度上评估自回归滚动。corrector 接收标准化当前状态。"""
    from ns_experiment.common import relative_l2, raw_mse

    model.eval()
    device = grid.device
    raw_states = raw_states.to(device)
    raw_forces = raw_forces.to(device)
    state_mean = state_mean.to(device)
    state_std = state_std.to(device)
    force_mean = force_mean.to(device)
    force_std = force_std.to(device)
    force = (raw_forces - force_mean) / force_std
    initial = (raw_states[:, 0] - state_mean) / state_std
    owners = torch.tensor(trajectory_ids, dtype=torch.long, device=device)

    def apply(current):
        return corrector(current, force, owners)

    prediction = rollout(
        model,
        initial,
        force,
        grid,
        steps,
        None if corrector is None else apply,
    )
    prediction = prediction * state_std + state_mean
    truth = raw_states[:, 1:steps + 1]
    names = ("velocity_x", "velocity_y", "particles")
    curves = []
    for step in range(steps):
        pred = prediction[:, step]
        target = truth[:, step]
        row = {
            "step": step + 1,
            "relative_l2": float(relative_l2(pred, target).mean()),
            "velocity_relative_l2": float(relative_l2(pred[..., :2], target[..., :2]).mean()),
            "mse": float(raw_mse(pred, target).mean()),
        }
        for index, name in enumerate(names):
            row[f"relative_l2_{name}"] = float(
                relative_l2(pred[..., index:index + 1], target[..., index:index + 1]).mean()
            )
        curves.append(row)
    summary = {"steps": curves}
    for key in curves[0]:
        if key == "step":
            continue
        summary[key] = float(sum(row[key] for row in curves) / len(curves))
    summary["final_relative_l2"] = curves[-1]["relative_l2"]
    tail = curves[-min(20, len(curves)):]
    summary["late_window_relative_l2"] = float(
        sum(row["relative_l2"] for row in tail) / len(tail)
    )
    return summary


@torch.no_grad()
def autoregressive_fields(
    model: nn.Module,
    raw_states: torch.Tensor,
    raw_forces: torch.Tensor,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    force_mean: torch.Tensor,
    force_std: torch.Tensor,
    grid: torch.Tensor,
    steps: int,
    trajectory_ids: list[int],
    corrector=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回原始尺度的预测和真值，形状 [B,steps,H,W,C]，位于 CPU。"""
    model.eval()
    device = grid.device
    raw_states = raw_states.to(device)
    raw_forces = raw_forces.to(device)
    state_mean = state_mean.to(device)
    state_std = state_std.to(device)
    force = (raw_forces - force_mean.to(device)) / force_std.to(device)
    initial = (raw_states[:, 0] - state_mean) / state_std
    owners = torch.tensor(trajectory_ids, dtype=torch.long, device=device)

    def apply(current):
        return corrector(current, force, owners)

    prediction = rollout(
        model,
        initial,
        force,
        grid,
        steps,
        None if corrector is None else apply,
    )
    prediction = (prediction * state_std + state_mean).cpu()
    truth = raw_states[:, 1:steps + 1].cpu()
    return prediction, truth
