"""生成外力强度不同的 NS 短轨迹，用于 CL 数据流程检查。"""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "4"

import argparse
import json
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phi.torch.flow import (
    Box, CenteredGrid, StaggeredGrid,
    extrapolation, fluid, advect, diffuse,
    math, spatial, channel, Solve,
)

ROOT = Path(__file__).resolve().parents[1]


def smooth_vector(rng, n, rms):
    """构造平滑向量场；按两个分量合并后的 RMS 定标。"""
    values = rng.normal(size=(n, n, 2))
    frequency = np.fft.fftfreq(n) * n
    kx, ky = np.meshgrid(frequency, frequency, indexing="ij")
    filt = np.exp(-(kx * kx + ky * ky) / 16.0)
    spectrum = np.fft.fft2(values, axes=(0, 1))
    values = np.fft.ifft2(
        spectrum * filt[..., None], axes=(0, 1)
    ).real
    values -= values.mean(axis=(0, 1), keepdims=True)
    values *= rms / max(float(np.sqrt(np.mean(values**2))), 1e-12)
    return values.astype(np.float32)


def vector_grid(values, bounds):
    return CenteredGrid(
        math.tensor(
            values,
            spatial(x=values.shape[0], y=values.shape[1]),
            channel(vector="x,y"),
        ),
        extrapolation.ZERO,
        bounds=bounds,
    )


def simulate(initial, force_values, dye, n, frames, dt=0.001):
    bounds = Box(x=1, y=1)
    template = StaggeredGrid(
        0, extrapolation.ZERO, x=n, y=n, bounds=bounds
    )
    velocity = vector_grid(initial, bounds) @ template
    force = vector_grid(force_values, bounds) @ template
    particles = CenteredGrid(
        math.tensor(dye, spatial(x=n, y=n)),
        extrapolation.BOUNDARY,
        bounds=bounds,
    )

    # 封闭边界下，显式指定压力零空间维数。
    solve = Solve(
        "CG",
        rel_tol=1e-8,
        abs_tol=1e-9,
        max_iterations=4000,
        rank_deficiency=1,
    )
    velocity, _ = fluid.make_incompressible(
        velocity, solve=solve
    )

    substeps = int(round(0.005 / dt))
    if dt <= 0 or substeps < 1 or abs(substeps * dt - 0.005) > 1e-12:
        raise ValueError("内部时间步必须整除保存间隔0.005。")
    viscosity = 0.01
    states, divergence_rms = [], []

    def record():
        vel = velocity.at_centers().values.numpy("x,y,vector")
        par = particles.values.numpy("x,y")
        state = np.concatenate([vel, par[..., None]], axis=-1)
        div = field_divergence(velocity)
        if not np.isfinite(state).all() or not np.isfinite(div):
            raise RuntimeError("生成结果出现 NaN/Inf，停止保存。")
        states.append(state.astype(np.float32))
        divergence_rms.append(div)

    record()
    for frame in range(1, frames):
        for _ in range(substeps):
            particles = advect.mac_cormack(
                particles, velocity, dt
            )
            velocity = advect.semi_lagrangian(
                velocity, velocity, dt
            )
            velocity = diffuse.explicit(
                velocity, viscosity, dt
            )
            velocity = velocity + force * dt
            velocity, _ = fluid.make_incompressible(
                velocity, solve=solve
            )
        record()
        if frame % 5 == 0:
            print(f"  已生成 {frame}/{frames - 1} 个时间间隔",
                  flush=True)

    return np.stack(states), divergence_rms


def field_divergence(velocity):
    from phi.torch.flow import field
    values = field.divergence(velocity).values.numpy("x,y")
    return float(np.sqrt(np.mean(values**2)))


def main():
    math.set_global_precision(64)
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--frames", type=int, default=21)
    args = parser.parse_args()

    n = args.resolution
    frames = args.frames
    if n < 8 or frames < 2:
        raise ValueError("分辨率至少8，帧数至少2。")

    output = ROOT / "data" / "cl_generated" / datetime.now().strftime(
        "force_shift_smoke_%Y%m%d_%H%M%S_%f"
    )
    output.mkdir(parents=True, exist_ok=False)

    rng = np.random.default_rng(10000)
    initial = smooth_vector(rng, n, rms=0.05)
    force_a = smooth_vector(rng, n, rms=0.15)
    force_b = 2.0 * force_a

    coordinate = (np.arange(n) + 0.5) / n
    x, y = np.meshgrid(coordinate, coordinate, indexing="ij")
    dye = np.exp(
        -((x - 0.5)**2 + (y - 0.5)**2) / 0.02
    ).astype(np.float32)

    config = {
        "purpose": "paired_short_trajectory_smoke_check",
        "is_pdebench_reproduction": False,
        "resolution": n,
        "frames": frames,
        "NU": 0.01,
        "internal_dt": 0.001,
        "substeps_per_frame": 5,
        "saved_dt": 0.005,
        "initial_vector_component_rms": 0.05,
        "force_vector_component_rms_A": 0.15,
        "force_vector_component_rms_B": 0.30,
        "seed": 10000,
        "pair_id": "10000",
        "state_channels": ["velocity_x", "velocity_y", "particles"],
        "phiflow": version("phiflow"),
        "phiml": version("phiml"),
        "torch": torch.__version__,
        "note": (
            "A/B share the initial state and force shape. "
            "Future train/valid/test splits must group by pair_id. "
            "CPU/GPU availability does not itself confirm solver placement."
        ),
    }

    states = {}
    diagnostics = {}
    print("输出目录：", output, flush=True)

    for task, force in (("A", force_a), ("B", force_b)):
        print(f"开始生成工况 {task}", flush=True)
        state, div = simulate(initial, force, dye, n, frames)
        states[task] = state
        diagnostics[task] = {
            "divergence_rms": div,
            "max_divergence_rms": max(div),
        }
        torch.save({
            "states": torch.from_numpy(state[None]),
            "forces": torch.from_numpy(force[None]),
            "times": torch.arange(frames, dtype=torch.float64)[None] * 0.005,
            "task": task,
            "pair_id": "10000",
            "config": config,
        }, output / f"task_{task}.pt")

    initial_gap = float(np.max(np.abs(states["A"][0] - states["B"][0])))
    final_gap = float(np.sqrt(np.mean(
        (states["A"][-1] - states["B"][-1])**2
    )))
    report = {
        "config": config,
        "diagnostics": diagnostics,
        "initial_state_max_abs_difference": initial_gap,
        "final_state_rms_difference": final_gap,
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    speed = {
        task: np.linalg.norm(state[-1, ..., :2], axis=-1)
        for task, state in states.items()
    }
    vmax = max(float(values.max()) for values in speed.values())
    fig, axes = plt.subplots(1, 2, figsize=(9, 4),
                             constrained_layout=True)
    for ax, task in zip(axes, ("A", "B")):
        image = ax.imshow(
            speed[task].T, origin="lower",
            extent=(0, 1, 0, 1),
            vmin=0, vmax=vmax, cmap="viridis",
        )
        ax.set_title(f"Task {task}: final speed")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.colorbar(image, ax=axes, label="Speed")
    fig.savefig(output / "task_comparison.png", dpi=180)
    plt.close(fig)

    (output / "summary.md").write_text(
        "# CL 新工况数据短轨迹检查\n\n"
        "- 本次只有一对轨迹，不能用于正式泛化评价。\n"
        "- A、B 使用相同初态与外力形状，B 外力幅度为 A 的两倍。\n"
        "- 两组使用相同网格、边界、黏性和积分方法。\n"
        "- 这不是 PDEBench 生成过程的精确复现。\n"
        f"- 初态最大差：{initial_gap:.6e}\n"
        f"- 末态 RMS 差：{final_gap:.6e}\n"
        f"- A 最大散度 RMS："
        f"{diagnostics['A']['max_divergence_rms']:.6e}\n"
        f"- B 最大散度 RMS："
        f"{diagnostics['B']['max_divergence_rms']:.6e}\n\n"
        "![末时刻速度对比](task_comparison.png)\n",
        encoding="utf-8",
    )
    print("生成完成：", output / "summary.md", flush=True)
    print("初态最大差：", initial_gap, flush=True)
    print("末态 RMS 差：", final_gap, flush=True)
    for task in ("A", "B"):
        print(
            task, "最大散度 RMS：",
            diagnostics[task]["max_divergence_rms"],
            flush=True,
        )



def generate_cohort():
    """固定配置的配对工况数据；按单条轨迹保存，可续跑。"""
    import time
    import sys
    import subprocess

    math.set_global_precision(64)
    torch.set_num_threads(4)

    output = ROOT / "data/cl_generated/force_shift_cohort_v1"
    output.mkdir(parents=True, exist_ok=True)

    config = {
        "version": 1,
        "purpose": "controlled_force_shift_CL_pilot",
        "is_pdebench_reproduction": False,
        "resolution": 64,
        "frames": 201,
        "internal_dt": 0.0005,
        "saved_dt": 0.005,
        "substeps_per_frame": 10,
        "NU": 0.01,
        "initial_vector_component_rms": 0.05,
        "force_vector_component_rms_A": 0.15,
        "force_vector_component_rms_B": 0.30,
        "pairs": 20,
        "seed_start": 10001,
        "split_seed": 0,
        "split_sizes_pairs": {"train": 12, "valid": 4, "test": 4},
        "solver_precision": 64,
        "state_storage_dtype": "float32",
        "phiflow": version("phiflow"),
        "phiml": version("phiml"),
        "torch": torch.__version__,
        "particle_initialization": "fixed_centered_gaussian",
    }

    config_path = output / "config.json"
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if old != config:
            raise RuntimeError("已有数据配置不同，不能混入同一目录。")
    else:
        config_path.write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )

    # 记录生成脚本，防止续跑时悄悄改变求解器实现。
    script_path = output / "generator_snapshot.py"
    current_script = Path(__file__).read_text(encoding="utf-8")
    if script_path.exists():
        if script_path.read_text(encoding="utf-8") != current_script:
            raise RuntimeError("生成脚本已改变，不能续写已有数据目录。")
    else:
        script_path.write_text(current_script, encoding="utf-8")

    environment = output / "environment.txt"
    if not environment.exists():
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True, check=True,
        )
        environment.write_text(result.stdout, encoding="utf-8")

    order = np.random.default_rng(0).permutation(20).tolist()
    split = {
        "train": order[:12],
        "valid": order[12:16],
        "test": order[16:],
    }

    n, frames = 64, 201
    coordinate = (np.arange(n) + 0.5) / n
    x, y = np.meshgrid(coordinate, coordinate, indexing="ij")
    dye = np.exp(
        -((x - 0.5)**2 + (y - 0.5)**2) / 0.02
    ).astype(np.float32)

    manifest = {
        "config": config,
        "split_pair_indices": split,
        "records": [],
    }
    start = time.perf_counter()

    def save_progress(status):
        manifest["status"] = status
        manifest["seconds_this_invocation"] = time.perf_counter() - start
        temporary = output / "manifest.json.tmp"
        temporary.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        temporary.replace(output / "manifest.json")

        lines = [
            "# 外力变化 CL 数据集",
            "",
            f"- 状态：{status}",
            f"- 已确认文件：{len(manifest['records'])}/40",
            "- 每个工况20条轨迹；训练/验证/测试为12/4/4。",
            "- 按pair划分，同一初态不会跨训练、验证、测试。",
            "- 空间64×64，201帧，保存间隔0.005，时间0到1。",
            "- 内部时间步0.0005，双精度求解，float32保存。",
            "- A、B共享初态和外力形状，仅外力幅度不同。",
            "- 这是自生成受控数据，不是PDEBench精确复现。",
            "- 相同网格可能存在空间离散误差。",
            "- 当前仍是小规模CL试验数据。",
            "- 完整配置、划分、散度与逐轨迹统计见manifest.json。",
        ]
        (output / "summary.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    print("数据目录：", output, flush=True)
    print("将生成A/B各20条轨迹，支持跳过已完成文件。", flush=True)

    try:
        for pair in range(20):
            seed = 10001 + pair
            rng = np.random.default_rng(seed)
            initial = smooth_vector(rng, n, rms=0.05)
            force_a = smooth_vector(rng, n, rms=0.15)
            group = next(
                name for name, ids in split.items() if pair in ids
            )
            pair_initial = {}

            for task, force in (("A", force_a), ("B", force_a * 2)):
                filename = f"task_{task}_pair_{pair:03d}.pt"
                destination = output / filename

                if destination.exists():
                    payload = torch.load(
                        destination, map_location="cpu",
                        weights_only=False,
                    )
                    if (
                        payload["config"] != config
                        or payload["seed"] != seed
                        or payload["task"] != task
                        or payload["split"] != group
                    ):
                        raise RuntimeError(f"已有文件元数据不一致：{filename}")
                    state = payload["states"][0].numpy()
                    div_max = payload["max_divergence_rms"]
                    print("复用：", filename, flush=True)
                else:
                    print(
                        f"生成 pair={pair + 1}/20，task={task}，split={group}",
                        flush=True,
                    )
                    state, divergence = simulate(
                        initial, force, dye, n, frames, dt=0.0005
                    )
                    div_max = float(max(divergence))
                    payload = {
                        "states": torch.from_numpy(state[None]),
                        "forces": torch.from_numpy(force[None]),
                        "times": (
                            torch.arange(frames, dtype=torch.float64)[None]
                            * 0.005
                        ),
                        "task": task,
                        "pair_id": str(seed),
                        "seed": seed,
                        "split": group,
                        "config": config,
                        "max_divergence_rms": div_max,
                        "divergence_rms": divergence,
                    }

                if state.shape != (201, 64, 64, 3):
                    raise RuntimeError(f"状态形状异常：{filename}")
                if not np.isfinite(state).all() or not np.isfinite(div_max):
                    raise RuntimeError(f"非有限值：{filename}")
                if not torch.isfinite(payload["forces"]).all().item():
                    raise RuntimeError(f"外力非有限值：{filename}")

                # 只做明显异常筛查，不把阈值当成精度证明。
                if div_max > 1e-5:
                    raise RuntimeError(
                        f"散度异常，停止并检查：{filename}, RMS={div_max}"
                    )

                pair_initial[task] = state[0].copy()

                if not destination.exists():
                    temporary = destination.with_suffix(".pt.tmp")
                    torch.save(payload, temporary)
                    temporary.replace(destination)

                manifest["records"].append({
                    "file": filename,
                    "task": task,
                    "pair_id": str(seed),
                    "split": group,
                    "max_divergence_rms": div_max,
                    "max_abs_state": float(np.max(np.abs(state))),
                })
                save_progress("running")

            gap = float(np.max(np.abs(
                pair_initial["A"] - pair_initial["B"]
            )))
            if gap > 1e-6:
                raise RuntimeError(f"配对初态不一致：pair={pair}, gap={gap}")

            print(
                f"完成配对 {pair + 1}/20，本次累计耗时"
                f" {time.perf_counter() - start:.1f} 秒",
                flush=True,
            )

        save_progress("completed")
        print("数据生成完成：", output / "summary.md", flush=True)

    except BaseException:
        save_progress("interrupted_or_failed")
        raise


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2 and sys.argv[1] == "--cohort":
        generate_cohort()
    else:
        main()