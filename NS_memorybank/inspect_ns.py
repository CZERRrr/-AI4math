"""检查 PDEBench 不可压 NS 的 HDF5，并把结论写入 results/。

截断文件不再在 h5py 打开时抛出未捕获异常。标准输出按行刷新，
同时写入 run.log、inspect_report.json 和 inspect_report.md。
不修改原始数据文件。
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass


def _repair_omp_threads() -> None:
    raw = os.environ.get("OMP_NUM_THREADS")
    if raw is None:
        return
    try:
        invalid = int(raw) < 1
    except ValueError:
        invalid = True
    if invalid:
        os.environ["OMP_NUM_THREADS"] = "1"
        print(
            f"OMP_NUM_THREADS={raw!r} 对 OpenMP 无效，本次检查改为 1。",
            flush=True,
        )


_repair_omp_threads()

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "ns_incom_inhom_2d_512-125.h5"
HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"

# 2026-10-18 从 DaRUS 文件记录核对，只用于和超块里的 stored EOF 对照。
PUBLISHED_FILE = {
    "source": "DaRUS file id 133444，数据集 doi:10.18419/DARUS-2986",
    "filename": "ns_incom_inhom_2d_512-125.h5",
    "size_bytes": 9970607580,
    "md5": "2b15572d1dda8048ab9713c2318d9322",
}

# PDEBench 生成器 4ff3e3a 的保存规则，用来解释 config，不能代替读出的 t。
GENERATOR_COMMIT = "4ff3e3a4aa1561721b5571fa3a048a0a463e0568"


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


def read_superblock(path: Path) -> dict:
    with path.open("rb") as handle:
        signature = handle.read(8)
        if signature != HDF5_SIGNATURE:
            raise ValueError(f"文件签名不是 HDF5：{signature!r}")
        version = struct.unpack("B", handle.read(1))[0]
        if version != 0:
            return {
                "version": version,
                "stored_eof": None,
                "eof_offset": None,
            }
        handle.seek(40)
        stored_eof = struct.unpack("<Q", handle.read(8))[0]
    return {
        "version": version,
        "stored_eof": int(stored_eof),
        "eof_offset": 40,
    }


def parse_config_text(text: str) -> dict:
    """读取生成器写进属性的扁平配置。嵌套的 dataverse 字段保留原文。"""
    values: dict[str, object] = {"raw": text}
    scalars = {
        "sim_name": str,
        "label": str,
        "particle_extrapolation": str,
        "velocity_extrapolation": str,
        "force_extrapolation": str,
        "NU": float,
        "seed": int,
        "smoothness": float,
        "scale": float,
        "force_smoothness": float,
        "force_scale": float,
        "n_steps": int,
        "DT": float,
        "frame_int": int,
        "n_batch": int,
    }
    for line in text.splitlines():
        if ":" not in line or line.startswith(" ") or line.startswith("-"):
            continue
        key, raw_value = line.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if key not in scalars or raw_value == "":
            continue
        caster = scalars[key]
        values[key] = caster(raw_value)
    return values


def infer_saved_times(config: dict, n_frames: int | None) -> dict:
    """按官方生成器的保存循环推断时间。t 数组未读出时，这只是推断。"""
    required = ("n_steps", "DT", "frame_int")
    if any(key not in config for key in required):
        return {"status": "config 缺少 n_steps、DT 或 frame_int"}

    n_steps = int(config["n_steps"])
    dt = float(config["DT"])
    frame_int = int(config["frame_int"])
    n_steps_actual = ((n_steps - 1) // frame_int) * frame_int + 1
    expected_frames = 1 + (n_steps_actual - 1) // frame_int
    save_dt = frame_int * dt
    report = {
        "status": "由 config 与生成器保存规则推断，不是从 t 数据集读出",
        "generator_commit": GENERATOR_COMMIT,
        "integration_dt": dt,
        "frame_interval_steps": frame_int,
        "saved_frame_dt": save_dt,
        "expected_frames_including_t0": expected_frames,
        "inferred_t0": 0.0,
        "inferred_t_last": (expected_frames - 1) * save_dt,
        "inferred_span": (expected_frames - 1) * save_dt,
        "matches_dataset_frame_count": (
            None if n_frames is None else expected_frames == n_frames
        ),
    }
    return report


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value


def open_for_inspection(path: Path, superblock: dict):
    """优先正常打开。截断时复制到临时文件并只改副本里的 EOF。"""
    actual_size = path.stat().st_size
    stored_eof = superblock.get("stored_eof")
    truncated = stored_eof is not None and stored_eof != actual_size
    try:
        return h5py.File(path, "r"), None, truncated
    except OSError as exc:
        if not truncated or superblock.get("eof_offset") is None:
            raise
        temporary = Path(tempfile.mkdtemp(prefix="ns_inspect_"))
        patched = temporary / "eof_patched_copy.h5"
        shutil.copyfile(path, patched)
        with patched.open("r+b") as handle:
            handle.seek(superblock["eof_offset"])
            handle.write(struct.pack("<Q", actual_size))
        print(
            "原始文件被 HDF5 判定为截断。已复制到临时文件，"
            "只修改副本超块中的 EOF 以读取对象头。",
            flush=True,
        )
        print(f"打开失败的原始错误：{exc}", flush=True)
        return h5py.File(patched, "r"), temporary, True


def probe_dataset(dataset: h5py.Dataset, file_size: int) -> dict:
    info = {
        "shape": list(dataset.shape),
        "dtype": str(dataset.dtype),
        "chunks": None if dataset.chunks is None else list(dataset.chunks),
        "compression": dataset.compression,
        "logical_nbytes": int(dataset.nbytes),
    }
    try:
        chunk_count = int(dataset.id.get_num_chunks())
        inside = 0
        for index in range(chunk_count):
            chunk = dataset.id.get_chunk_info(index)
            end = int(chunk.byte_offset) + int(chunk.size)
            if 0 <= int(chunk.byte_offset) and end <= file_size:
                inside += 1
        info["n_chunks"] = chunk_count
        info["chunks_inside_local_file"] = inside
    except Exception as exc:
        info["chunk_index_error"] = f"{type(exc).__name__}: {exc}"

    try:
        if dataset.nbytes <= 64 * 1024 * 1024:
            values = np.asarray(dataset[...])
            info["payload"] = "complete"
        else:
            values = np.asarray(dataset[0])
            info["payload"] = "first_index_only"
        finite = bool(np.isfinite(values).all())
        info["readable"] = True
        info["finite"] = finite
        info["sample_shape"] = list(values.shape)
        info["min"] = float(np.min(values))
        info["max"] = float(np.max(values))
        if dataset.ndim == 2 and dataset.shape[1] <= 10000:
            info["row0_head"] = np.asarray(values[0, :8], dtype=np.float64).tolist()
            info["row0_tail"] = np.asarray(values[0, -4:], dtype=np.float64).tolist()
            delta = np.diff(np.asarray(values, dtype=np.float64), axis=1)
            info["strictly_increasing_per_row"] = bool(np.all(delta > 0))
            info["rows_identical"] = bool(
                np.allclose(values, values[0:1], rtol=0, atol=0)
            )
    except Exception as exc:
        info["readable"] = False
        info["read_error"] = f"{type(exc).__name__}: {exc}"
    return info


def markdown_report(report: dict) -> str:
    file_info = report["file"]
    lines = [
        "# NS HDF5 检查",
        "",
        f"- 检查时间：{report['checked_at']}",
        f"- 状态：{report['status']}",
        f"- 数据路径：`{file_info['path']}`",
        f"- 本地大小：{file_info['size_bytes']} 字节"
        f"（{file_info['size_gib']:.6f} GiB）",
        f"- HDF5 超块版本：{file_info['superblock_version']}",
        f"- 超块记录的完整 EOF：{file_info['stored_eof_bytes']}",
        f"- 是否截断：{file_info['truncated']}",
        f"- 与 DaRUS 公布大小一致："
        f"{file_info['stored_eof_matches_published_size']}",
        f"- DaRUS 对照：{PUBLISHED_FILE['source']}，"
        f"MD5 `{PUBLISHED_FILE['md5']}`",
        "",
        "## 字段",
        "",
    ]
    for name, info in report["datasets"].items():
        lines.append(
            f"- `{name}`：shape={info.get('shape')}，dtype={info.get('dtype')}，"
            f"compression={info.get('compression')}，"
            f"可读={info.get('readable')}"
        )
        if info.get("read_error"):
            lines.append(f"  - 读取错误：{info['read_error']}")
        if info.get("chunks_inside_local_file") is not None:
            lines.append(
                "  - 落在本地文件内的数据块："
                f"{info['chunks_inside_local_file']}/{info.get('n_chunks')}"
            )
        if info.get("finite") is not None:
            lines.append(
                f"  - 有限值={info['finite']}，"
                f"min={info.get('min')}，max={info.get('max')}"
            )
    lines.extend(["", "## 轨迹与时间", ""])
    for key, value in report["trajectory"].items():
        lines.append(f"- {key}：{value}")
    lines.extend(["", "## 物理条件", ""])
    for key, value in report["condition"].items():
        if key == "config_raw":
            continue
        lines.append(f"- {key}：{value}")
    lines.extend(
        [
            "",
            "## 能否进入训练",
            "",
            report["training_decision"],
            "",
            "## 尚未验证",
            "",
        ]
    )
    for item in report["not_verified"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def inspect(path: Path) -> dict:
    actual_size = path.stat().st_size
    superblock = read_superblock(path)
    stored_eof = superblock.get("stored_eof")
    truncated = stored_eof is not None and int(stored_eof) != actual_size
    print("=" * 60, flush=True)
    print("文件：", path, flush=True)
    print(
        "本地大小：",
        actual_size,
        f"字节（{actual_size / 1024 ** 3:.6f} GiB）",
        flush=True,
    )
    print("超块记录的 EOF：", stored_eof, flush=True)
    print("文件是否截断：", truncated, flush=True)

    temporary = None
    datasets: dict[str, dict] = {}
    condition: dict[str, object] = {}
    trajectory: dict[str, object] = {}
    time_read = None
    try:
        handle, temporary, _ = open_for_inspection(path, superblock)
        with handle:
            print("顶层字段：", list(handle.keys()), flush=True)
            print("文件属性：", list(handle.attrs.keys()), flush=True)
            if "config" in handle.attrs:
                config_text = handle.attrs["config"]
                if isinstance(config_text, bytes):
                    config_text = config_text.decode("utf-8", "replace")
                condition = parse_config_text(str(config_text))
                condition["config_raw"] = str(config_text)
                print("config 属性已读出。", flush=True)
            if "latestIndex" in handle.attrs:
                condition["latestIndex"] = int(handle.attrs["latestIndex"])
                print("latestIndex：", condition["latestIndex"], flush=True)

            for name in handle.keys():
                print(f"\n检查字段 {name}", flush=True)
                info = probe_dataset(handle[name], actual_size)
                datasets[name] = info
                print(
                    f"shape={info.get('shape')} dtype={info.get('dtype')} "
                    f"readable={info.get('readable')}",
                    flush=True,
                )
                if not info.get("readable"):
                    print(info.get("read_error"), flush=True)

            if "t" in datasets and datasets["t"].get("readable"):
                time_values = np.asarray(handle["t"][...], dtype=np.float64)
                time_read = {
                    "shape": list(time_values.shape),
                    "t0": float(time_values[0, 0]),
                    "t_last": float(time_values[0, -1]),
                    "first_dt": float(time_values[0, 1] - time_values[0, 0]),
                    "strictly_increasing": bool(
                        np.all(np.diff(time_values, axis=1) > 0)
                    ),
                    "all_rows_equal": bool(
                        np.allclose(time_values, time_values[0:1])
                    ),
                }
                print(
                    "已读出 t："
                    f"{time_read['t0']} 到 {time_read['t_last']}，"
                    f"首间隔 {time_read['first_dt']}",
                    flush=True,
                )
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)

    velocity = datasets.get("velocity", {})
    particles = datasets.get("particles", {})
    force = datasets.get("force", {})
    time_info = datasets.get("t", {})
    counts = []
    for info in (velocity, particles, force, time_info):
        shape = info.get("shape")
        if shape:
            counts.append(int(shape[0]))
    unique_counts = sorted(set(counts))
    n_frames = None
    if velocity.get("shape") and len(velocity["shape"]) >= 2:
        n_frames = int(velocity["shape"][1])
    inferred = infer_saved_times(condition, n_frames)

    payloads_ready = all(
        datasets.get(name, {}).get("readable")
        for name in ("velocity", "particles", "t", "force")
    )
    same_count = len(unique_counts) == 1
    trajectory.update(
        {
            "leading_dimensions": {
                name: info.get("shape", [None])[0]
                for name, info in datasets.items()
            },
            "independent_trajectory_count": (
                unique_counts[0] if same_count else None
            ),
            "leading_dimensions_agree": same_count,
            "frames_per_trajectory": n_frames,
            "spatial": (
                velocity.get("shape")[2:4] if velocity.get("shape") else None
            ),
            "velocity_components": (
                velocity.get("shape")[-1] if velocity.get("shape") else None
            ),
            "time_array_read": time_read,
            "time_inferred_from_generator": inferred,
        }
    )

    if payloads_ready and same_count and unique_counts[0] >= 4:
        decision = (
            f"四个字段的数据都能读取，独立轨迹数为 {unique_counts[0]}。"
            "可以按完整轨迹划分 train/valid/test。"
            "4 条只能做 2/1/1 的冒烟划分，不能把它写成大样本泛化结果。"
        )
        status = "ready"
    elif payloads_ready and same_count and unique_counts[0] < 4:
        decision = (
            f"字段可读，但独立轨迹只有 {unique_counts[0]} 条。"
            "不足 4 条，不能做 train/valid/test 三组，"
            "也不要把同一条轨迹的相邻帧拆进三个集合。"
        )
        status = "too_few_trajectories"
    else:
        decision = (
            "velocity、particles 或 t 的数据块不在本地文件中，"
            "不能训练，也不能制造基于相邻帧的划分。"
            "force 即使可读，也只是静态外力，不是可滚动的状态轨迹。"
            "需要把文件补到超块记录的 EOF 字节数后再检查。"
        )
        status = "truncated"

    not_verified = []
    if time_read is None:
        not_verified.append(
            "t 数据集的数值还没有读出；报告中的时间范围来自生成器规则推断。"
        )
    not_verified.append("本脚本只检查数据文件，不判断已有训练结果。")
    if not payloads_ready:
        not_verified.append(
            "速度和 particles 的取值范围、有限性和逐轨迹时间轴都还未核对。"
        )

    print("\n【结论】", flush=True)
    print(decision, flush=True)
    print("独立轨迹领先维：", unique_counts or "未能识别", flush=True)
    return {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "file": {
            "path": str(path),
            "size_bytes": actual_size,
            "size_gib": actual_size / 1024 ** 3,
            "superblock_version": superblock.get("version"),
            "stored_eof_bytes": stored_eof,
            "truncated": truncated,
            "stored_eof_matches_published_size": (
                stored_eof == PUBLISHED_FILE["size_bytes"]
            ),
            "published_file": PUBLISHED_FILE,
        },
        "datasets": datasets,
        "condition": {
            key: value
            for key, value in condition.items()
            if key != "raw"
        },
        "trajectory": trajectory,
        "training_decision": decision,
        "not_verified": not_verified,
    }


def _panel(axis, field, vmin, vmax, cmap, title):
    image = axis.imshow(np.asarray(field).T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_title(title, fontsize=9)
    axis.set_xticks([])
    axis.set_yticks([])
    return image


def save_dataset_figures(path: Path, run_dir: Path) -> list[str]:
    """只读取少量原始帧，画出轨迹、外力和初值差异。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    saved = []
    with h5py.File(path, "r") as handle:
        count = int(handle["velocity"].shape[0])
        frames = int(handle["velocity"].shape[1])
        times = np.asarray(handle["t"][0], dtype=np.float64)
        wanted = [0, min(100, frames - 1), min(500, frames - 1), frames - 1]
        frame_ids = list(dict.fromkeys(wanted))
        velocity = np.stack([np.asarray(handle["velocity"][0, index]) for index in frame_ids])
        particles = np.stack([np.asarray(handle["particles"][0, index, ..., 0]) for index in frame_ids])
        initial_velocity = np.stack([np.asarray(handle["velocity"][index, 0]) for index in range(count)])
        initial_particles = np.stack([
            np.asarray(handle["particles"][index, 0, ..., 0]) for index in range(count)
        ])
        force = np.asarray(handle["force"][:])

    speed = np.sqrt(velocity[..., 0] ** 2 + velocity[..., 1] ** 2)
    rows = [
        ("vx", velocity[..., 0], "RdBu_r", True),
        ("vy", velocity[..., 1], "RdBu_r", True),
        ("speed", speed, "viridis", False),
        ("particles", particles, "viridis", False),
    ]
    figure, axes = plt.subplots(
        len(rows), len(frame_ids), figsize=(3.0 * len(frame_ids), 10.5), constrained_layout=True
    )
    for row, (name, values, cmap, symmetric) in enumerate(rows):
        limit = float(np.max(np.abs(values))) if symmetric else None
        vmin = -limit if symmetric else float(values.min())
        vmax = limit if symmetric else float(values.max())
        image = None
        for column, frame in enumerate(frame_ids):
            image = _panel(axes[row, column], values[column], vmin, vmax, cmap, f"{name}  t={times[frame]:.3f}")
        figure.colorbar(image, ax=axes[row, :], fraction=0.046, pad=0.02)
    figure.suptitle("Trajectory 0 at 512x512, before downsampling")
    evolution = run_dir / "dataset_trajectory0.png"
    figure.savefig(evolution, dpi=140)
    plt.close(figure)
    saved.append(evolution.name)

    initial_speed = np.sqrt(initial_velocity[..., 0] ** 2 + initial_velocity[..., 1] ** 2)
    figure, axes = plt.subplots(2, count, figsize=(3.0 * count, 6.2), constrained_layout=True)
    speed_limit = float(initial_speed.max())
    particle_limit = float(np.max(np.abs(initial_particles)))
    for index in range(count):
        image_speed = _panel(
            axes[0, index], initial_speed[index], 0, speed_limit, "viridis", f"traj {index} speed"
        )
        image_particles = _panel(
            axes[1, index],
            initial_particles[index],
            -particle_limit,
            particle_limit,
            "RdBu_r",
            f"traj {index} particles",
        )
    figure.colorbar(image_speed, ax=axes[0, :], fraction=0.046, pad=0.02)
    figure.colorbar(image_particles, ax=axes[1, :], fraction=0.046, pad=0.02)
    figure.suptitle(f"Four initial states at t={float(times[0]):.3f}")
    initial_path = run_dir / "dataset_initial_states.png"
    figure.savefig(initial_path, dpi=140)
    plt.close(figure)
    saved.append(initial_path.name)

    figure, axes = plt.subplots(2, count, figsize=(3.0 * count, 6.2), constrained_layout=True)
    force_limit = float(np.max(np.abs(force)))
    for index in range(count):
        _panel(axes[0, index], force[index, ..., 0], -force_limit, force_limit, "RdBu_r", f"traj {index} fx")
        image_force = _panel(
            axes[1, index], force[index, ..., 1], -force_limit, force_limit, "RdBu_r", f"traj {index} fy"
        )
    figure.colorbar(image_force, ax=axes, fraction=0.046, pad=0.02)
    figure.suptitle("Static force field for each trajectory")
    force_path = run_dir / "dataset_force.png"
    figure.savefig(force_path, dpi=140)
    plt.close(figure)
    saved.append(force_path.name)
    print("数据图像：", ", ".join(saved), flush=True)
    return saved


def main() -> int:
    run_dir = ROOT / "results" / datetime.now().strftime(
        "inspect_ns_%Y%m%d_%H%M%S"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    log_handle = (run_dir / "run.log").open("w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_handle)
    sys.stderr = _Tee(sys.__stderr__, log_handle)
    print("检查记录目录：", run_dir, flush=True)

    if "--figures" in sys.argv:
        if not DATA_PATH.is_file():
            print(f"找不到数据文件：{DATA_PATH}", flush=True)
            return 1
        figures = save_dataset_figures(DATA_PATH, run_dir)
        (run_dir / "dataset_figures.md").write_text(
            "# NS 数据图像\n\n"
            "这些图直接来自 HDF5 的原始 512×512 场，还没有做训练用的块平均。\n\n"
            + "\n".join(f"- `{name}`" for name in figures)
            + "\n",
            encoding="utf-8",
        )
        print("数据图像目录：", run_dir, flush=True)
        return 0

    if not DATA_PATH.is_file():
        print(f"找不到数据文件：{DATA_PATH}", flush=True)
        return 1

    try:
        report = inspect(DATA_PATH)
    except Exception:
        traceback.print_exc()
        print("检查因未预期错误中断。", flush=True)
        return 1

    (run_dir / "inspect_report.json").write_text(
        json.dumps(_jsonable(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "inspect_report.md").write_text(
        markdown_report(report),
        encoding="utf-8",
    )
    print("JSON：", run_dir / "inspect_report.json", flush=True)
    print("Markdown：", run_dir / "inspect_report.md", flush=True)
    if report["status"] == "ready":
        figures = save_dataset_figures(DATA_PATH, run_dir)
        report_path = run_dir / "inspect_report.md"
        report_path.write_text(
            report_path.read_text(encoding="utf-8").rstrip()
            + "\n## 图像\n\n"
            + "\n".join(f"- `{name}`" for name in figures)
            + "\n",
            encoding="utf-8",
        )
    print("数据结构检查完成。", flush=True)
    return 0 if report["status"] == "ready" else 2


if __name__ == "__main__":
    sys.exit(main())
