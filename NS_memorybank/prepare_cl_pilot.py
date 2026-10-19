from pathlib import Path
import hashlib
import json
import time

import numpy as np
import torch
from ns_experiment import generate_cl_data as g

g.math.set_global_precision(64)
torch.set_num_threads(4)

source = g.ROOT / "data/cl_generated/force_shift_cohort_v1"
output = g.ROOT / "data/cl_generated/force_shift_pilot_v1"
output.mkdir(parents=True, exist_ok=True)

split = {"train": [0, 1, 2, 3], "valid": [4], "test": [5]}
script = Path(g.__file__).read_bytes()
snapshot = source / "generator_snapshot.py"
if not snapshot.exists() or snapshot.read_bytes() != script:
    raise RuntimeError("生成器与原始数据的脚本快照不同，请停止检查。")

config = {
    "version": "pilot_v1",
    "pairs": 6,
    "frames": 101,
    "resolution": 64,
    "NU": 0.01,
    "internal_dt": 0.0005,
    "saved_dt": 0.005,
    "substeps_per_frame": 10,
    "solver_precision": 64,
    "force_vector_component_rms_A": 0.15,
    "force_vector_component_rms_B": 0.30,
    "seed_start": 10001,
    "split": split,
    "generator_sha256": hashlib.sha256(script).hexdigest(),
    "is_pdebench_reproduction": False,
}

config_path = output / "config.json"
if config_path.exists():
    if json.loads(config_path.read_text()) != config:
        raise RuntimeError("已有 pilot 配置不同，不能混写。")
else:
    config_path.write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

(output / "generator_snapshot.py").write_bytes(script)

n, frames = 64, 101
coordinate = (np.arange(n) + 0.5) / n
x, y = np.meshgrid(coordinate, coordinate, indexing="ij")
dye = np.exp(
    -((x - 0.5)**2 + (y - 0.5)**2) / 0.02
).astype(np.float32)
clock = torch.arange(frames, dtype=torch.float64)[None] * 0.005

records = []
start = time.perf_counter()


def progress(status):
    report = {
        "status": status,
        "config": config,
        "records": records,
        "seconds_this_invocation": time.perf_counter() - start,
    }
    temporary = output / "manifest.json.tmp"
    temporary.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    temporary.replace(output / "manifest.json")
    (output / "summary.md").write_text(
        "# 外力变化 CL 小规模数据\n\n"
        f"- 状态：{status}\n"
        f"- 已确认文件：{len(records)}/12\n"
        "- 每个工况 6 条轨迹，训练/验证/测试为 4/1/1。\n"
        "- A/B 按配对初态划分，同一初态不跨数据集。\n"
        "- 空间 64×64，101 帧，时间 0～0.5。\n"
        "- A/B 外力幅度为 0.15/0.30。\n"
        "- 自生成受控试验，不是 PDEBench 精确复现。\n",
        encoding="utf-8",
    )


print("输出目录：", output, flush=True)

try:
    for pair in range(6):
        seed = 10001 + pair
        group = next(k for k, ids in split.items() if pair in ids)
        rng = np.random.default_rng(seed)
        initial = g.smooth_vector(rng, n, rms=0.05)
        force_a = g.smooth_vector(rng, n, rms=0.15)
        initial_states = {}

        for task, force in (("A", force_a), ("B", force_a * 2)):
            name = f"task_{task}_pair_{pair:03d}.pt"
            destination = output / name
            original = source / name

            if destination.exists():
                payload = torch.load(
                    destination, map_location="cpu", weights_only=False
                )
                if (
                    payload["config"] != config
                    or payload["seed"] != seed
                    or payload["task"] != task
                    or payload["split"] != group
                ):
                    raise RuntimeError(f"已有文件配置不符：{name}")
                print("已完成，跳过生成：", name, flush=True)

            else:
                if original.exists():
                    old = torch.load(
                        original, map_location="cpu", weights_only=False
                    )
                    required = {
                        "resolution": 64,
                        "NU": 0.01,
                        "internal_dt": 0.0005,
                        "saved_dt": 0.005,
                        "solver_precision": 64,
                    }
                    if (
                        old["seed"] != seed
                        or old["task"] != task
                        or any(
                            old["config"].get(k) != v
                            for k, v in required.items()
                        )
                        or old["states"].shape[1] < frames
                    ):
                        raise RuntimeError(f"源文件配置不符：{name}")
                    if not torch.allclose(
                        old["times"][:, :frames], clock
                    ):
                        raise RuntimeError(f"源文件时间不符：{name}")
                    if not torch.allclose(
                        old["forces"], torch.from_numpy(force[None])
                    ):
                        raise RuntimeError(f"源文件外力不符：{name}")

                    states = old["states"][:, :frames].clone()
                    div_max = float(old["max_divergence_rms"])
                    provenance = str(original)
                    print("裁剪已有数据：", name, flush=True)

                else:
                    print(
                        f"开始补生成：pair={pair + 1}/6，"
                        f"task={task}，split={group}",
                        flush=True,
                    )
                    state, divergence = g.simulate(
                        initial, force, dye, n, frames, dt=0.0005
                    )
                    states = torch.from_numpy(state[None])
                    div_max = float(max(divergence))
                    provenance = "generated_directly_101_frames"

                payload = {
                    "states": states,
                    "forces": torch.from_numpy(force[None]),
                    "times": clock.clone(),
                    "task": task,
                    "pair_id": str(seed),
                    "seed": seed,
                    "split": group,
                    "config": config,
                    "max_divergence_rms": div_max,
                    "provenance": provenance,
                }

            states = payload["states"]
            div_max = float(payload["max_divergence_rms"])
            if states.shape != (1, 101, 64, 64, 3):
                raise RuntimeError(f"状态形状异常：{name}")
            if (
                not torch.isfinite(states).all().item()
                or not torch.isfinite(payload["forces"]).all().item()
                or not np.isfinite(div_max)
                or div_max > 1e-5
            ):
                raise RuntimeError(f"数据质量检查失败：{name}")
            if not torch.allclose(payload["times"], clock):
                raise RuntimeError(f"时间坐标异常：{name}")
            if not torch.allclose(
                payload["forces"], torch.from_numpy(force[None])
            ):
                raise RuntimeError(f"外力异常：{name}")

            initial_states[task] = states[:, 0].clone()
            if task == "B":
                gap = float(
                    (initial_states["A"] - initial_states["B"])
                    .abs().max()
                )
                if gap > 1e-6:
                    raise RuntimeError(f"配对初态不一致：{name}")

            if not destination.exists():
                temporary = destination.with_suffix(".pt.tmp")
                torch.save(payload, temporary)
                temporary.replace(destination)

            records.append({
                "file": name,
                "task": task,
                "pair_id": str(seed),
                "split": group,
                "max_divergence_rms": div_max,
                "provenance": payload["provenance"],
            })
            progress("running")
            print(f"已保存并检查：{len(records)}/12", flush=True)

    progress("completed")
    print("数据准备完成：", output / "summary.md", flush=True)

except BaseException:
    progress("interrupted_or_failed")
    raise
