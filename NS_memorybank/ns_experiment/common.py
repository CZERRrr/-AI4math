"""实验目录、指标和检查点。训练脚本共用，不读取 SWE 数据。"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


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


class RunLog:
    def __init__(self, run_dir: Path):
        run_dir.mkdir(parents=True, exist_ok=False)
        self.run_dir = run_dir
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        self._handle = (run_dir / "run.log").open("w", encoding="utf-8")
        sys.stdout = _Tee(self._stdout, self._handle)
        sys.stderr = _Tee(self._stderr, self._handle)

    def close(self) -> None:
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        self._handle.close()


def write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"不能写入 JSON：{type(value)!r}")


def write_markdown(path: Path, title: str, sections: list[tuple[str, list[str]]]) -> None:
    lines = [f"# {title}", ""]
    for heading, bullets in sections:
        lines.append(f"## {heading}")
        lines.append("")
        for bullet in bullets:
            lines.append(f"- {bullet}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def relative_l2(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """按 batch 计算原始尺度 Relative L2。"""
    difference = (prediction - truth).to(torch.float64).flatten(1)
    target = truth.to(torch.float64).flatten(1)
    numerator = torch.linalg.vector_norm(difference, dim=1)
    denominator = torch.linalg.vector_norm(target, dim=1).clamp_min(1e-12)
    return numerator / denominator


def raw_mse(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    return (prediction - truth).to(torch.float64).flatten(1).square().mean(dim=1)


def summarize_fields(prediction: torch.Tensor, truth: torch.Tensor) -> dict[str, float]:
    """prediction、truth：[B,H,W,C]。联合误差之外单独给速度和各通道。"""
    joint = relative_l2(prediction, truth)
    velocity = relative_l2(prediction[..., :2], truth[..., :2])
    names = ("velocity_x", "velocity_y", "particles")
    report = {
        "relative_l2": float(joint.mean()),
        "velocity_relative_l2": float(velocity.mean()),
        "mse": float(raw_mse(prediction, truth).mean()),
    }
    for index, name in enumerate(names):
        report[f"relative_l2_{name}"] = float(
            relative_l2(prediction[..., index:index + 1], truth[..., index:index + 1]).mean()
        )
    return report


def parameter_count(module: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    return total, trainable
