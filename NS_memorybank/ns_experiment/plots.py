"""把曲线和物理场画成 PNG。图中文字用英文，避免服务器缺少中文字体。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def append_figure_notes(summary_path: Path, filenames: list[str]) -> None:
    text = summary_path.read_text(encoding="utf-8") if summary_path.is_file() else "# 图像\n"
    if "## 图像" in text:
        return
    lines = ["", "## 图像", ""]
    lines.extend(f"- `{name}`" for name in filenames)
    summary_path.write_text(text.rstrip() + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def save_training_curves(history: list[dict], path: Path) -> None:
    epochs = [int(row["epoch"]) for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(epochs, [float(row["train_normalized_mse"]) for row in history], marker="o")
    axes[0].set_title("Train one-step normalized MSE")
    axes[0].set_ylabel("MSE")
    axes[1].plot(
        epochs,
        [float(row["valid_relative_l2"]) for row in history],
        marker="o",
        label="joint",
    )
    if "valid_velocity_relative_l2" in history[0]:
        axes[1].plot(
            epochs,
            [float(row["valid_velocity_relative_l2"]) for row in history],
            marker="o",
            label="velocity",
        )
        axes[1].legend()
    axes[1].set_title("Validation 100-step rollout")
    axes[1].set_ylabel("Relative L2")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def save_budget_curves(history: list[dict], path: Path) -> None:
    """预算实验的训练曲线：单步误差，以及验证集 1/10/100 步。"""
    epochs = [int(row["epoch"]) for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(epochs, [float(row["train_relative_l2"]) for row in history], marker="o")
    axes[0].set_title("Train one-step Relative L2")
    axes[0].set_ylabel("Relative L2")
    for key, label in (
        ("valid_l2_1", "1 step"),
        ("valid_l2_10", "10 steps"),
        ("valid_l2_100", "100 steps"),
    ):
        axes[1].plot(epochs, [float(row[key]) for row in history], marker="o", label=label)
    axes[1].set_title("Validation rollout, unseen trajectories, t=0 to 0.5")
    axes[1].set_ylabel("Relative L2")
    axes[1].legend()
    for axis in axes:
        axis.set_xlabel("Epoch (20 updates)")
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def save_rollout_curves(steps: list[dict], path: Path, title: str, mark_step: int | None = None) -> None:
    x = [int(row["step"]) for row in steps]
    time = [float(row["time"]) for row in steps] if "time" in steps[0] else None
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(x, [float(row["relative_l2"]) for row in steps], label="joint")
    axes[0].plot(x, [float(row["velocity_relative_l2"]) for row in steps], label="velocity")
    axes[0].plot(x, [float(row["relative_l2_particles"]) for row in steps], label="particles")
    axes[0].set_ylabel("Relative L2")
    axes[0].legend()
    axes[1].plot(x, [float(row["relative_l2_velocity_x"]) for row in steps], label="vx")
    axes[1].plot(x, [float(row["relative_l2_velocity_y"]) for row in steps], label="vy")
    axes[1].set_ylabel("Relative L2")
    axes[1].legend()
    for axis in axes:
        axis.set_xlabel("Forecast step" if time is None else "Forecast step (t marked below)")
        axis.grid(alpha=0.3)
        axis.axhline(1.0, color="0.5", lw=0.8, ls="--")
        if mark_step is not None:
            axis.axvline(mark_step, color="0.35", lw=0.8, ls=":")
    if time is not None:
        axes[0].set_title(f"{title}\nt={time[0]:.3f} to {time[-1]:.3f}")
    else:
        axes[0].set_title(title)
    axes[1].set_title("Velocity components")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def save_router_curves(history: list[dict], path: Path) -> None:
    epochs = [int(row["epoch"]) for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(epochs, [float(row["train_mse"]) for row in history], marker="o")
    axes[0].set_title("Router multi-step normalized MSE")
    axes[0].set_ylabel("MSE")
    axes[1].plot(epochs, [float(row["valid_relative_l2_gamma1"]) for row in history], marker="o")
    axes[1].set_title("Validation rollout at gamma=1")
    axes[1].set_ylabel("Relative L2")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def save_method_fields(
    path: Path,
    truth: np.ndarray,
    predictions: list[tuple[str, np.ndarray]],
    times: list[float],
    indices: list[int],
) -> None:
    """truth 和每个预测都是 [steps,H,W]。行是时间，列是真值和各方法。"""
    columns = [("truth", truth)] + predictions
    selected = [[field[index] for index in indices] for _name, field in columns]
    limit = float(max(np.max(np.abs(np.stack(row))) for row in selected))
    figure, axes = plt.subplots(
        len(indices),
        len(columns),
        figsize=(2.8 * len(columns), 2.8 * len(indices)),
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(len(indices), len(columns))
    image = None
    for row, index in enumerate(indices):
        for column, (name, _field) in enumerate(columns):
            image = _show(
                axes[row, column],
                selected[column][row],
                -limit,
                limit,
                "RdBu_r",
                f"{name}  t={times[index]:.3f}",
            )
    figure.colorbar(image, ax=axes, fraction=0.03, pad=0.02)
    figure.suptitle("Speed on the 64x64 test rollout")
    figure.savefig(path, dpi=140)
    plt.close(figure)


def save_method_bars(rows: list[dict], path: Path) -> None:
    labels = [row["method"] for row in rows]
    keys = [
        ("relative_l2", "joint"),
        ("velocity_relative_l2", "velocity"),
        ("relative_l2_particles", "particles"),
    ]
    x = np.arange(len(labels))
    width = 0.25
    figure, axis = plt.subplots(figsize=(8, 4))
    for index, (key, name) in enumerate(keys):
        axis.bar(x + (index - 1) * width, [float(row[key]) for row in rows], width, label=name)
    axis.set_xticks(x, labels)
    axis.set_ylabel("Mean rollout Relative L2")
    axis.set_title("Same test trajectory and forecast window")
    axis.axhline(1.0, color="0.5", lw=0.8, ls="--")
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def save_gamma_search(fixed_rows: list[dict], learned_rows: list[dict], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(
        [float(row["gamma"]) for row in fixed_rows],
        [float(row["valid_relative_l2"]) for row in fixed_rows],
        marker="o",
        label="fixed retrieval",
    )
    axis.plot(
        [float(row["gamma"]) for row in learned_rows],
        [float(row["valid_relative_l2"]) for row in learned_rows],
        marker="o",
        label="trained router",
    )
    axis.set_xlabel("gamma")
    axis.set_ylabel("Validation rollout Relative L2")
    axis.set_title("Gamma chosen on the validation trajectory")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _show(axis, field: np.ndarray, vmin: float, vmax: float, cmap: str, title: str):
    image = axis.imshow(field.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_title(title, fontsize=9)
    axis.set_xticks([])
    axis.set_yticks([])
    return image


def save_sequence(path: Path, frames: list[tuple[str, np.ndarray]], cmap: str, symmetric: bool) -> None:
    """frames 是同一色标下的若干二维场。"""
    stacked = np.stack([field for _title, field in frames])
    if symmetric:
        limit = float(np.max(np.abs(stacked)))
        vmin, vmax = -limit, limit
    else:
        vmin, vmax = float(stacked.min()), float(stacked.max())
    columns = len(frames)
    figure, axes = plt.subplots(1, columns, figsize=(3.1 * columns, 3.2), constrained_layout=True)
    if columns == 1:
        axes = [axes]
    image = None
    for axis, (title, field) in zip(axes, frames):
        image = _show(axis, field, vmin, vmax, cmap, title)
    figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def save_truth_prediction(
    path: Path,
    truth: np.ndarray,
    prediction: np.ndarray,
    times: list[float],
    indices: list[int],
    quantity: str,
) -> None:
    """truth、prediction：[steps,H,W]。每个时间一列，三行是真值、预测、绝对误差。"""
    selected_truth = [truth[index] for index in indices]
    selected_pred = [prediction[index] for index in indices]
    limit = float(max(np.max(np.abs(selected_truth)), np.max(np.abs(selected_pred))))
    error_limit = float(max(np.max(np.abs(selected_pred[i] - selected_truth[i])) for i in range(len(indices))))
    figure, axes = plt.subplots(3, len(indices), figsize=(3.0 * len(indices), 8.2), constrained_layout=True)
    axes = np.asarray(axes).reshape(3, len(indices))
    for column, index in enumerate(indices):
        title = f"t={times[index]:.3f}"
        _show(axes[0, column], selected_truth[column], -limit, limit, "RdBu_r", f"truth {title}")
        image = _show(axes[1, column], selected_pred[column], -limit, limit, "RdBu_r", f"prediction {title}")
        error = _show(
            axes[2, column],
            selected_pred[column] - selected_truth[column],
            -error_limit,
            error_limit,
            "RdBu_r",
            "prediction - truth",
        )
    figure.colorbar(image, ax=axes[:2, :], fraction=0.03, pad=0.02)
    figure.colorbar(error, ax=axes[2, :], fraction=0.03, pad=0.02)
    figure.suptitle(quantity)
    figure.savefig(path, dpi=140)
    plt.close(figure)
