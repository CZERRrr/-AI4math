from pathlib import Path
from datetime import datetime
import csv
import json
import random
import time

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib

# 只保存图片，不弹出窗口
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ffno_core.grid_2d import FNOFactorized2DBlock


# ============================================================
# 1. 路径与配置
# ============================================================

# 本文件位于 SWE_MemoryBank/baseline_ffno/
ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "2D_rdb_NA_NA.h5"

CFG = {
    "seed": 0,
    "ntrain": 800,
    "nvalid": 100,
    "ntest": 100,
    "epochs": 300,
    "batch_size":32,
    "learning_rate": 1e-3,
    "downsample": 2,
    "width": 32,
    "modes": 16,
    "layers": 4,
    "factor": 4,
    "device": "cpu",
}

RUN_DIR = (
    ROOT
    / "results"
    / datetime.now().strftime("ffno_swe_%Y%m%d_%H%M%S_%f")
)


# ============================================================
# 2. 读取数据，按完整轨迹划分
# ============================================================

def load_data():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"没有找到数据文件：{DATA_PATH}")

    for name in ("ntrain", "nvalid", "ntest", "epochs", "batch_size"):
        if CFG[name] < 1:
            raise ValueError(f"{name} 必须至少为 1。")

    with h5py.File(DATA_PATH, "r") as f:
        keys = sorted([
            key
            for key in f.keys()
            if isinstance(f[key], h5py.Group)
            and "data" in f[key]
            and isinstance(f[key]["data"], h5py.Dataset)
        ])

        # 固定随机种子，便于重复实验
        rng = np.random.default_rng(CFG["seed"])
        rng.shuffle(keys)

        required = CFG["ntrain"] + CFG["nvalid"] + CFG["ntest"]

        if required > len(keys):
            raise ValueError(
                f"需要 {required} 条轨迹，但文件中只有 {len(keys)} 条。"
            )

        keys = keys[:required]

        trajectories = []
        reference_t = None
        reference_shape = None

        for index, key in enumerate(keys, start=1):
            ds = f[key]["data"]

            if ds.ndim != 4 or ds.shape[-1] != 1:
                raise ValueError(
                    f"轨迹 {key} 形状异常：{ds.shape}，"
                    "预期为 [T,H,W,1]。"
                )

            h = np.asarray(ds[..., 0], dtype=np.float32)

            t, nx, ny = h.shape
            r = CFG["downsample"]

            if r < 1 or nx % r != 0 or ny % r != 0:
                raise ValueError("下采样倍数必须为正整数并整除空间尺寸。")

            # 对 r×r 空间小块取平均，保留全部时间帧
            h = h.reshape(
                t, nx // r, r, ny // r, r
            ).mean(axis=(2, 4))

            if not np.isfinite(h).all():
                raise ValueError(f"轨迹 {key} 存在 NaN 或 Inf。")

            times = np.asarray(
                f[key]["grid"]["t"][...],
                dtype=np.float64,
            )

            if len(times) != t or not np.all(np.diff(times) > 0):
                raise ValueError(f"轨迹 {key} 的时间坐标异常。")

            if reference_t is None:
                reference_t = times
                reference_shape = h.shape
            else:
                if h.shape != reference_shape:
                    raise ValueError("所选轨迹的数据形状不一致。")

                if (
                    times.shape != reference_t.shape
                    or not np.allclose(times, reference_t)
                ):
                    raise ValueError("所选轨迹的时间网格不一致。")

            trajectories.append(h)

            if index % 10 == 0 or index == required:
                print(
                    f"已读取 {index}/{required} 条轨迹",
                    flush=True,
                )

    # [N,T,H,W,1]，官方模型使用通道在最后的格式
    data = torch.from_numpy(
        np.stack(trajectories)
    ).unsqueeze(-1)

    if data.shape[1] < 2:
        raise ValueError("至少需要两个时间帧。")

    ntrain = CFG["ntrain"]
    valid_end = ntrain + CFG["nvalid"]

    train = data[:ntrain]
    valid = data[ntrain:valid_end]
    test = data[valid_end:]

    split_ids = {
        "train": keys[:ntrain],
        "valid": keys[ntrain:valid_end],
        "test": keys[valid_end:],
    }

    return train, valid, test, split_ids, reference_t


class OneStepDataset(Dataset):
    """从训练轨迹中提取相邻帧：h_t -> h_(t+1)。"""

    def __init__(self, normalized_trajectories):
        self.data = normalized_trajectories
        self.steps = self.data.shape[1] - 1

    def __len__(self):
        return len(self.data) * self.steps

    def __getitem__(self, index):
        trajectory_id, t = divmod(index, self.steps)

        x = self.data[trajectory_id, t]
        y = self.data[trajectory_id, t + 1]

        return x, y


# ============================================================
# 3. 官方 FFNO 模型
# ============================================================

def build_model(device):
    model = FNOFactorized2DBlock(
        modes=CFG["modes"],
        width=CFG["width"],
        input_dim=3,  # 当前水深 + x坐标 + y坐标
        n_layers=CFG["layers"],
        factor=CFG["factor"],
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


def make_grid(nx, ny, device):
    """生成归一化到 [0,1] 的位置坐标。"""
    x = torch.linspace(0, 1, nx, device=device)
    y = torch.linspace(0, 1, ny, device=device)

    xx, yy = torch.meshgrid(x, y, indexing="ij")

    # [1,H,W,2]
    return torch.stack([xx, yy], dim=-1).unsqueeze(0)


def predict(model, h, grid):
    """
    h: [B,H,W,1]，标准化后的当前水深。
    返回标准化后的下一帧水深。
    """
    coordinates = grid.expand(h.shape[0], -1, -1, -1)

    # [B,H,W,3]
    inputs = torch.cat([h, coordinates], dim=-1)

    return model(inputs)["forecast"]


def synchronize(device):
    """GPU 计时时等待计算完成；CPU 无需同步。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


# ============================================================
# 4. 指标与自回归评估
# ============================================================

def relative_l2(pred, truth):
    """双精度计算指标，不改变模型本身的 float32 训练。"""
    pred64 = pred.to(torch.float64)
    truth64 = truth.to(torch.float64)

    difference = (pred64 - truth64).flatten(1)
    target = truth64.flatten(1)

    numerator = torch.linalg.vector_norm(difference, dim=1)
    denominator = torch.linalg.vector_norm(
        target, dim=1
    ).clamp_min(1e-12)

    return numerator / denominator


@torch.no_grad()
def evaluate_rollout(model, data, mean, std, grid, device):
    model.eval()

    horizon = data.shape[1] - 1
    l2_sum = np.zeros(horizon, dtype=np.float64)
    mse_sum = np.zeros(horizon, dtype=np.float64)

    for batch_index, raw in enumerate(
        data.split(CFG["batch_size"]), start=1
    ):
        raw = raw.to(device)
        current = (raw[:, 0] - mean) / std

        for step in range(horizon):
            current = predict(model, current, grid)

            if not torch.isfinite(current).all():
                raise RuntimeError(
                    f"rollout 第 {batch_index} 个 batch、"
                    f"第 {step + 1} 步："
                    "模型输出出现 NaN/Inf，预测已数值发散。"
                )

            # 在双精度下反标准化并计算指标
            pred = current.to(torch.float64) * std + mean
            truth = raw[:, step + 1].to(torch.float64)

            batch_l2 = relative_l2(pred, truth)
            batch_mse = (
                (pred - truth)
                .square()
                .flatten(1)
                .mean(dim=1)
            )

            # 只打印第一个 batch，避免输出过多
            if batch_index == 1 and (
                step == 0 or (step + 1) % 10 == 0
            ):
                print(
                    f"  rollout step {step + 1:03d} | "
                    f"pred=[{pred.min().item():.3e}, "
                    f"{pred.max().item():.3e}] | "
                    f"truth=[{truth.min().item():.3e}, "
                    f"{truth.max().item():.3e}] | "
                    f"L2={batch_l2.mean().item():.3e}",
                    flush=True,
                )

            if (
                not torch.isfinite(batch_l2).all()
                or not torch.isfinite(batch_mse).all()
            ):
                raise RuntimeError(
                    f"rollout 第 {batch_index} 个 batch、"
                    f"第 {step + 1} 步：双精度指标仍然溢出。"
                )

            l2_sum[step] += batch_l2.sum().item()
            mse_sum[step] += batch_mse.sum().item()

    l2_curve = l2_sum / len(data)
    mse_curve = mse_sum / len(data)

    return {
        "l2": float(l2_curve.mean()),
        "mse": float(mse_curve.mean()),
        "l2_curve": l2_curve,
        "mse_curve": mse_curve,
    }

# ============================================================
# 5. Summary 与图片
# ============================================================

def save_reports(history, test_result, summary):
    with open(
        RUN_DIR / "summary.json", "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    sections = {
        "Model Information": {
            "Total parameters": summary["total_parameters"],
            "Trainable parameters": summary["trainable_parameters"],
        },

        "Training Configuration": {
            "dataset": DATA_PATH.name,
            **CFG,
            "Tin": 1,
            "Tout": summary["rollout_steps"],
            "optimizer": "Adam",
            "training_loss": "MSE on normalized water depth",
            "evaluation_metric": "Relative L2 and MSE on raw water depth",
            "training_target": "one-step prediction",
            "checkpoint_selection":
                "minimum validation rollout Relative L2",
        },

        "Additional Information": {
            "model": "Official FNOFactorized2DBlock",
            "experiment_type": "SWE adaptation, no memory",
            "spatial_resolution": summary["spatial_resolution"],
            "normalization": "train-only global mean and std",
            "prediction": "direct next-state prediction",
            "boundary_handling": "original spectral model, no padding",
            "metric_aggregation":
                "mean over trajectories and forecast steps",
            "train_metric_note":
                "online training-batch metrics, not fixed-checkpoint evaluation",
            "best_model": "best_model.pt",
            "loss_curve": "loss_curve.png",
        },

        "Best Validation Result": {
            "Best epoch": summary["best_epoch"],
            "Best Validation Rollout Relative L2":
                summary["best_valid_l2"],
        },

        "Final Test Result": {
            "Test Rollout MSE": summary["test_mse"],
            "Test Rollout Relative L2": summary["test_l2"],
            "Test Final-step Relative L2":
                summary["test_final_step_l2"],
        },

        "Final Training Epoch": {
            "Final Train MSE (raw units, one-step)":
                history[-1]["train_mse"],
            "Final Train Relative L2 (one-step)":
                history[-1]["train_l2"],
        },

        "Runtime": {
            "Total training time including validation/checkpoints (s)":
                summary["train_seconds"],
            "Average epoch time (s)":
                summary["average_epoch_seconds"],
            "Test evaluation time (s)":
                summary["test_seconds"],
        },
    }

    lines = ["# Experiment: FFNO Baseline on SWE", ""]

    for title, values in sections.items():
        lines.extend([f"## {title}", ""])

        for key, value in values.items():
            if isinstance(value, float):
                value = f"{value:.8f}"
            lines.append(f"- {key}: {value}")

        lines.append("")

    (RUN_DIR / "summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    epochs = [row["epoch"] for row in history]

    # 分开画，避免把单步训练误差和 rollout 验证误差混为一谈
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(
        epochs,
        [row["train_mse"] for row in history],
        marker="o",
    )
    axes[0].set_title("Train one-step MSE")
    axes[0].set_ylabel("MSE (raw water depth)")

    axes[1].plot(
        epochs,
        [row["valid_l2"] for row in history],
        marker="o",
    )
    axes[1].set_title("Validation rollout")
    axes[1].set_ylabel("Relative L2")

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(RUN_DIR / "loss_curve.png", dpi=180)
    plt.close(fig)

    # 测试集误差随预测步数的变化
    steps = np.arange(1, len(test_result["l2_curve"]) + 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, test_result["l2_curve"])
    ax.set_xlabel("Forecast step")
    ax.set_ylabel("Relative L2")
    ax.set_title("Test autoregressive rollout")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(RUN_DIR / "test_rollout_curve.png", dpi=180)
    plt.close(fig)


# ============================================================
# 6. 主训练流程
# ============================================================

def main():
    random.seed(CFG["seed"])
    np.random.seed(CFG["seed"])
    torch.manual_seed(CFG["seed"])

    device = torch.device(CFG["device"])
    RUN_DIR.mkdir(parents=True, exist_ok=False)

    train, valid, test, split_ids, times = load_data()

    with open(RUN_DIR / "split_ids.json", "w") as f:
        json.dump(split_ids, f, indent=2)

    # 只从训练集计算标准化参数
    mean = float(train.mean())
    std = max(float(train.std()), 1e-6)

    dataset = OneStepDataset((train - mean) / std)

    loader = DataLoader(
        dataset,
        batch_size=CFG["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    model = build_model(device)

    nx, ny = train.shape[2:4]
    if CFG["modes"] > min(nx // 2 + 1, ny // 2 + 1):
        raise ValueError("modes 超过当前网格允许的频率数量。")

    grid = make_grid(nx, ny, device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CFG["learning_rate"],
    )

    total_parameters = sum(
        p.numel() for p in model.parameters()
    )
    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\n========== 实验设置 ==========")
    print("设备：", device)
    print(
        "train / valid / test：",
        len(train), len(valid), len(test),
    )
    print("单条轨迹形状：", tuple(train.shape[1:]))
    print("训练相邻帧对数量：", len(dataset))
    print("每个 epoch 的 batch 数：", len(loader))
    print("总参数量：", total_parameters)
    print("Epochs：", CFG["epochs"])
    print("结果目录：", RUN_DIR, flush=True)

    history = []
    best_valid = float("inf")
    best_epoch = 0

    synchronize(device)
    training_start = time.perf_counter()

    fields = [
        "epoch",
        "train_normalized_mse",
        "train_mse",
        "train_l2",
        "valid_mse",
        "valid_l2",
        "seconds",
    ]

    with open(
        RUN_DIR / "history.csv", "w", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for epoch in range(1, CFG["epochs"] + 1):
            synchronize(device)
            epoch_start = time.perf_counter()

            model.train()
            mse_sum = 0.0
            l2_sum = 0.0
            count = 0

            for batch_index, (x, y) in enumerate(loader, start=1):
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad(set_to_none=True)

                pred = predict(model, x, grid)
                loss = (pred - y).square().mean()

                if not torch.isfinite(loss):
                    raise RuntimeError("训练损失出现 NaN/Inf。")

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                    error_if_nonfinite=True,
                )
                optimizer.step()

                batch_size = len(x)
                mse_sum += loss.item() * batch_size

                with torch.no_grad():
                    l2_sum += relative_l2(
                        pred * std + mean,
                        y * std + mean,
                    ).sum().item()

                count += batch_size

                if batch_index % 100 == 0:
                    print(
                        f"Epoch {epoch}/{CFG['epochs']} | "
                        f"batch {batch_index}/{len(loader)}",
                        flush=True,
                    )

            print(
                f"Epoch {epoch}：开始验证 100 步左右的 rollout...",
                flush=True,
            )

            val = evaluate_rollout(
                model, valid, mean, std, grid, device
            )

            if val["l2"] < best_valid:
                best_valid = val["l2"]
                best_epoch = epoch

                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": CFG,
                        "mean": mean,
                        "std": std,
                        "epoch": epoch,
                        "valid_l2": best_valid,
                    },
                    RUN_DIR / "best_model.pt",
                )

            synchronize(device)
            seconds = time.perf_counter() - epoch_start

            row = {
                "epoch": epoch,
                "train_normalized_mse": mse_sum / count,
                "train_mse": mse_sum / count * std ** 2,
                "train_l2": l2_sum / count,
                "valid_mse": val["mse"],
                "valid_l2": val["l2"],
                "seconds": seconds,
            }

            history.append(row)
            writer.writerow(row)
            f.flush()

            print(
                f"Epoch {epoch:03d}/{CFG['epochs']} | "
                f"train MSE={row['train_mse']:.8f} | "
                f"train L2={row['train_l2']:.6f} | "
                f"valid rollout L2={row['valid_l2']:.6f} | "
                f"time={seconds:.1f}s",
                flush=True,
            )

    synchronize(device)
    train_seconds = time.perf_counter() - training_start

    # 训练结束后加载验证集最优模型
    checkpoint = torch.load(
        RUN_DIR / "best_model.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model"])

    print("\n使用最佳模型评估测试集...", flush=True)

    synchronize(device)
    test_start = time.perf_counter()

    test_result = evaluate_rollout(
        model, test, mean, std, grid, device
    )

    synchronize(device)
    test_seconds = time.perf_counter() - test_start

    with open(
        RUN_DIR / "test_rollout.csv", "w", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "forecast_step", "time", "relative_l2", "mse"
        ])

        for i, (l2, mse) in enumerate(
            zip(
                test_result["l2_curve"],
                test_result["mse_curve"],
            ),
            start=1,
        ):
            writer.writerow([
                i, float(times[i]), float(l2), float(mse)
            ])

    summary = {
        "config": CFG,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "spatial_resolution": [nx, ny],
        "rollout_steps": int(train.shape[1] - 1),
        "normalization_mean": mean,
        "normalization_std": std,
        "best_epoch": best_epoch,
        "best_valid_l2": best_valid,
        "test_mse": test_result["mse"],
        "test_l2": test_result["l2"],
        "test_final_step_l2":
            float(test_result["l2_curve"][-1]),
        "train_seconds": train_seconds,
        "average_epoch_seconds": float(
            np.mean([row["seconds"] for row in history])
        ),
        "test_seconds": test_seconds,
    }

    save_reports(history, test_result, summary)

    print("\n========== 训练完成 ==========")
    print("最佳 epoch：", best_epoch)
    print(f"最佳 valid rollout L2：{best_valid:.8f}")
    print(f"Test rollout MSE：{test_result['mse']:.8f}")
    print(f"Test rollout L2：{test_result['l2']:.8f}")
    print(f"训练与验证总时间：{train_seconds:.2f} s")
    print("Summary：", RUN_DIR / "summary.md")


if __name__ == "__main__":
    main()