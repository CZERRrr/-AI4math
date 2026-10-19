"""读取 PDEBench 非均匀不可压 NS，并按完整轨迹划分。

只接受本地大小与 HDF5 超块 EOF 一致的文件。截断文件直接失败，
不用相邻帧假装成独立测试轨迹。
"""

from __future__ import annotations

import struct
from pathlib import Path

import h5py
import numpy as np
import torch

HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
STATE_CHANNELS = ("velocity_x", "velocity_y", "particles")


def stored_eof(path: Path) -> int:
    with path.open("rb") as handle:
        if handle.read(8) != HDF5_SIGNATURE:
            raise ValueError(f"{path} 不是 HDF5。")
        version = handle.read(1)[0]
        if version != 0:
            raise ValueError(f"尚未支持的 HDF5 超块版本：{version}")
        handle.seek(40)
        return struct.unpack("<Q", handle.read(8))[0]


def assert_complete(path: Path) -> int:
    actual = path.stat().st_size
    expected = stored_eof(path)
    if actual != expected:
        raise RuntimeError(
            f"{path} 不完整：本地 {actual} 字节，超块 EOF {expected} 字节。"
            "不能用缺速度场的文件训练。"
        )
    return actual


def block_average(array: np.ndarray, factor: int) -> np.ndarray:
    if factor < 1:
        raise ValueError("下采样倍数必须为正整数。")
    if factor == 1:
        return array
    *lead, height, width, channels = array.shape
    if height % factor != 0 or width % factor != 0:
        raise ValueError(
            f"下采样倍数 {factor} 不能整除空间尺寸 {(height, width)}。"
        )
    folded = array.reshape(
        *lead,
        height // factor,
        factor,
        width // factor,
        factor,
        channels,
    )
    return folded.mean(axis=(-4, -2)).astype(np.float32, copy=False)


def _parse_config(text: str) -> dict:
    values: dict = {"raw": text}
    casters = {
        "sim_name": str,
        "label": str,
        "NU": float,
        "seed": int,
        "n_steps": int,
        "DT": float,
        "frame_int": int,
        "n_batch": int,
        "particle_extrapolation": str,
        "velocity_extrapolation": str,
        "force_extrapolation": str,
        "smoothness": float,
        "scale": float,
        "force_smoothness": float,
        "force_scale": float,
    }
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.endswith(":") and line[:1] not in {" ", "-"}:
            key = line[:-1].strip()
            items = []
            cursor = index + 1
            while cursor < len(lines) and lines[cursor].startswith("- "):
                items.append(lines[cursor][2:].strip())
                cursor += 1
            if key in {"grid_size", "domain_size"} and items:
                caster = int if key == "grid_size" else float
                values[key] = [caster(item) for item in items]
                index = cursor
                continue
        if ":" in line and line[:1] not in {" ", "-"}:
            key, raw_value = line.split(":", 1)
            key = key.strip()
            raw_value = raw_value.strip()
            if key in casters and raw_value:
                values[key] = casters[key](raw_value)
        index += 1
    return values


# seed 可以不同。这些物理和离散设置必须一致，才并入同一次训练。
SAME_CONDITION = {
    "label": "ns_incomp_inhom_2d",
    "NU": 0.01,
    "particle_extrapolation": "BOUNDARY",
    "velocity_extrapolation": "ZERO",
    "force_extrapolation": "ZERO",
    "smoothness": 1.0,
    "scale": 0.4,
    "force_smoothness": 3.0,
    "force_scale": 0.15,
    "n_steps": 100000,
    "DT": 5e-5,
    "frame_int": 100,
    "grid_size": [512, 512],
    "domain_size": [1.0, 1.0],
}
COHORT_SPLITS = {16: (10, 3, 3), 20: (12, 4, 4)}


def split_trajectory_indices(count: int, seed: int) -> dict:
    """按完整轨迹划分。4 条只能是 2/1/1。"""
    if count < 4:
        raise RuntimeError(
            f"独立轨迹只有 {count} 条，不能划分 train/valid/test。"
        )
    order = np.random.default_rng(seed).permutation(count)
    if count == 4:
        sizes = (2, 1, 1)
        rule = "独立轨迹恰好 4 条，使用 2/1/1。这不是 70/15/15，也不能当作大样本泛化。"
    else:
        train = max(1, int(round(count * 0.70)))
        valid = max(1, int(round(count * 0.15)))
        test = count - train - valid
        if test < 1:
            train -= 1
            test = 1
        sizes = (train, valid, test)
        rule = "按约 70/15/15 划分，每组至少 1 条完整轨迹。"
    train_ids = order[:sizes[0]].astype(int).tolist()
    valid_ids = order[sizes[0]:sizes[0] + sizes[1]].astype(int).tolist()
    test_ids = order[sizes[0] + sizes[1]:].astype(int).tolist()
    if len(test_ids) != sizes[2]:
        raise RuntimeError("轨迹划分数量不一致。")
    return {
        "train": train_ids,
        "valid": valid_ids,
        "test": test_ids,
        "rule": rule,
        "seed": seed,
        "count": count,
    }


def load_ns_bundle(path: Path, downsample: int, max_frames: int) -> dict:
    """下采样后的原始尺度数据，归一化留给调用方按训练集计算。"""
    path = path.resolve()
    size_bytes = assert_complete(path)
    with h5py.File(path, "r") as handle:
        required = ("velocity", "particles", "force", "t")
        missing = [name for name in required if name not in handle]
        if missing:
            raise RuntimeError(f"缺少字段 {missing}，实际字段为 {list(handle.keys())}。")
        config_text = handle.attrs.get("config", "")
        if isinstance(config_text, bytes):
            config_text = config_text.decode("utf-8", "replace")
        config = _parse_config(str(config_text))
        latest = handle.attrs.get("latestIndex")
        velocity = handle["velocity"]
        particles = handle["particles"]
        force = handle["force"]
        time_ds = handle["t"]
        if velocity.shape[0] != particles.shape[0] or velocity.shape[0] != force.shape[0]:
            raise RuntimeError(
                "velocity、particles、force 的轨迹维不一致："
                f"{velocity.shape} {particles.shape} {force.shape}"
            )
        if velocity.shape[-1] != 2 or particles.shape[-1] != 1 or force.shape[-1] != 2:
            raise RuntimeError("通道数与 PDEBench NS 布局不一致。")
        count = int(velocity.shape[0])
        available = int(velocity.shape[1])
        source_resolution = [int(velocity.shape[2]), int(velocity.shape[3])]
        frames = min(int(max_frames), available)
        if frames < 2:
            raise RuntimeError("可用帧数不足 2。")
        print(
            f"读取 {count} 条轨迹，每条使用前 {frames}/{available} 帧，"
            f"空间 {velocity.shape[2]}×{velocity.shape[3]} 块平均到 "
            f"{velocity.shape[2] // downsample}×{velocity.shape[3] // downsample}。",
            flush=True,
        )
        states = []
        forces = []
        times = []
        for index in range(count):
            parts = []
            for start in range(0, frames, 25):
                end = min(frames, start + 25)
                vel = np.asarray(velocity[index, start:end], dtype=np.float32)
                par = np.asarray(particles[index, start:end], dtype=np.float32)
                parts.append(block_average(np.concatenate([vel, par], axis=-1), downsample))
                print(f"  轨迹 {index}：已处理 {end}/{frames} 帧", flush=True)
            state = np.concatenate(parts, axis=0)
            if not np.isfinite(state).all():
                raise RuntimeError(f"轨迹 {index} 的状态下采样后存在 NaN/Inf。")
            body = block_average(np.asarray(force[index], dtype=np.float32), downsample)
            if not np.isfinite(body).all():
                raise RuntimeError(f"轨迹 {index} 的外力存在 NaN/Inf。")
            clock = np.asarray(time_ds[index, :frames], dtype=np.float64)
            if clock.shape[0] != frames or not np.all(np.diff(clock) > 0):
                raise RuntimeError(f"轨迹 {index} 的时间坐标不是严格递增。")
            states.append(state)
            forces.append(body)
            times.append(clock)
    time_array = np.stack(times)
    if not np.allclose(time_array, time_array[0:1]):
        raise RuntimeError("各轨迹的时间坐标不一致，不能共用一个预测步长。")
    return {
        "path": str(path),
        "size_bytes": size_bytes,
        "states": torch.from_numpy(np.stack(states)),
        "forces": torch.from_numpy(np.stack(forces)),
        "times": torch.from_numpy(time_array),
        "config": config,
        "latest_index_attr": None if latest is None else int(latest),
        "source_frames": available,
        "used_frames": frames,
        "source_resolution": source_resolution,
        "downsample": int(downsample),
        "channels": list(STATE_CHANNELS),
        "channel_order_note": (
            "速度两个分量来自 PDEBench to_ndarray 的 vector 维，"
            "按 PhiFlow 空间维顺序记为 velocity_x、velocity_y；"
            "第三通道是数据集中的 particles。没有用解析解另行核对方向。"
        ),
    }


def population_mean_std(values: torch.Tensor, dims: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    mean = values.mean(dim=dims)
    std = values.std(dim=dims, unbiased=False).clamp_min(1e-6)
    return mean, std


def take_split(bundle: dict, indices: list[int]) -> dict:
    selector = torch.tensor(indices, dtype=torch.long)
    return {
        "indices": list(indices),
        "states": bundle["states"].index_select(0, selector),
        "forces": bundle["forces"].index_select(0, selector),
        "times": bundle["times"].index_select(0, selector),
    }


def normalize_state(states: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    view = mean.view(*([1] * (states.ndim - 1)), -1)
    scale = std.view(*([1] * (states.ndim - 1)), -1)
    return (states - view) / scale


def normalize_force(force: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    view = mean.view(*([1] * (force.ndim - 1)), -1)
    scale = std.view(*([1] * (force.ndim - 1)), -1)
    return (force - view) / scale


def condition_differences(config: dict) -> list[str]:
    differences = []
    for key, expected in SAME_CONDITION.items():
        actual = config.get(key)
        if isinstance(expected, float):
            if actual is None or abs(float(actual) - expected) > 1e-8:
                differences.append(f"{key}: {actual} != {expected}")
        elif isinstance(expected, list):
            if list(actual or []) != list(expected):
                differences.append(f"{key}: {actual} != {expected}")
        elif actual != expected:
            differences.append(f"{key}: {actual} != {expected}")
    return differences


def inspect_source(path: Path) -> dict:
    """检查完整性、配置、时间轴和轨迹数，不把速度场读进内存。"""
    path = path.resolve()
    actual = path.stat().st_size
    eof = stored_eof(path)
    report = {
        "path": str(path),
        "name": path.name,
        "size_bytes": actual,
        "stored_eof": eof,
        "complete": actual == eof,
    }
    if not report["complete"]:
        report["error"] = "文件大小和 HDF5 超块 EOF 不一致。"
        return report
    with h5py.File(path, "r") as handle:
        missing = [name for name in ("velocity", "particles", "force", "t") if name not in handle]
        if missing:
            report["error"] = f"缺少字段 {missing}。"
            return report
        config_text = handle.attrs.get("config", "")
        if isinstance(config_text, bytes):
            config_text = config_text.decode("utf-8", "replace")
        config = _parse_config(str(config_text))
        velocity, particles, force, time_ds = (
            handle["velocity"], handle["particles"], handle["force"], handle["t"]
        )
        times = np.asarray(time_ds[:], dtype=np.float64)
        report.update(
            {
                "config": {key: value for key, value in config.items() if key != "raw"},
                "latest_index_attr": (
                    None if "latestIndex" not in handle.attrs else int(handle.attrs["latestIndex"])
                ),
                "shapes": {
                    "velocity": list(velocity.shape),
                    "particles": list(particles.shape),
                    "force": list(force.shape),
                    "t": list(time_ds.shape),
                },
                "trajectory_count": int(velocity.shape[0]),
                "source_frames": int(velocity.shape[1]),
                "source_resolution": [int(velocity.shape[2]), int(velocity.shape[3])],
            }
        )
    if times.ndim != 2 or times.shape[0] != report["trajectory_count"]:
        report["error"] = f"时间坐标形状异常：{list(times.shape)}。"
        return report
    delta = np.diff(times, axis=1)
    report["time"] = {
        "t0": float(times[0, 0]),
        "t_last": float(times[0, -1]),
        "first_dt": float(delta[0, 0]),
        "strictly_increasing": bool(np.all(delta > 0)),
        "rows_identical": bool(np.allclose(times, times[0:1])),
        "finite": bool(np.isfinite(times).all()),
    }
    report["condition_differences"] = condition_differences(report["config"])
    report["same_condition"] = not report["condition_differences"] and report["time"]["strictly_increasing"]
    return report


def cache_path_for(cache_dir: Path, source: Path, frames: int, downsample: int) -> Path:
    return cache_dir / f"{source.stem}_down{downsample}_f{frames}.pt"


def build_file_cache(path: Path, cache_dir: Path, frames: int = 201, downsample: int = 8) -> Path:
    """把一个完整文件的前 frames 帧收成 64×64 缓存。不计算归一化。"""
    path = path.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_path_for(cache_dir, path, frames, downsample)
    inspection = inspect_source(path)
    if not inspection.get("complete"):
        raise RuntimeError(inspection.get("error", f"{path} 不完整。"))
    if not inspection.get("same_condition"):
        raise RuntimeError(f"{path.name} 与 NU=0.01 的参考工况不一致：{inspection.get('condition_differences')}")
    if inspection["source_frames"] < frames:
        raise RuntimeError(f"{path.name} 只有 {inspection['source_frames']} 帧，评估需要 {frames} 帧。")
    bundle = load_ns_bundle(path, downsample, frames)
    identities = []
    config = bundle["config"]
    for index in range(bundle["states"].shape[0]):
        identities.append(
            {
                "trajectory_key": f"{path.name}#{index}",
                "source_file": str(path),
                "source_name": path.name,
                "index_in_file": index,
                "seed": config.get("seed"),
                "NU": config.get("NU"),
                "particle_extrapolation": config.get("particle_extrapolation"),
                "velocity_extrapolation": config.get("velocity_extrapolation"),
                "force_extrapolation": config.get("force_extrapolation"),
                "smoothness": config.get("smoothness"),
                "scale": config.get("scale"),
                "force_smoothness": config.get("force_smoothness"),
                "force_scale": config.get("force_scale"),
                "grid_size": config.get("grid_size"),
                "domain_size": config.get("domain_size"),
                "DT": config.get("DT"),
                "frame_int": config.get("frame_int"),
                "n_steps": config.get("n_steps"),
            }
        )
    payload = {
        "states": bundle["states"],
        "forces": bundle["forces"],
        "times": bundle["times"],
        "identities": identities,
        "source_file": str(path),
        "source_size_bytes": bundle["size_bytes"],
        "source_frames": bundle["source_frames"],
        "used_frames": bundle["used_frames"],
        "source_resolution": bundle["source_resolution"],
        "downsample": downsample,
        "channels": bundle["channels"],
        "config": {key: value for key, value in config.items() if key != "raw"},
        "time_check": inspection["time"],
    }
    torch.save(payload, destination)
    print(f"缓存：{destination} ({destination.stat().st_size} 字节)", flush=True)
    return destination


def load_cached_trajectories(cache_files: list[Path]) -> list[dict]:
    """展开缓存。同一源文件或同一 trajectory_key 重复出现时直接失败。"""
    records = []
    seen_files = set()
    seen_keys = set()
    seen_seeds = set()
    for cache_file in cache_files:
        payload = torch.load(cache_file, map_location="cpu", weights_only=False)
        source = str(Path(payload["source_file"]).resolve())
        if source in seen_files:
            raise RuntimeError(f"缓存重复包含源文件 {source}。")
        seen_files.add(source)
        seed = payload["config"].get("seed")
        if seed in seen_seeds:
            raise RuntimeError(f"seed={seed} 出现在多个文件中，不能把重复轨迹放进划分。")
        seen_seeds.add(seed)
        count = payload["states"].shape[0]
        for index in range(count):
            identity = payload["identities"][index]
            key = identity["trajectory_key"]
            if key in seen_keys:
                raise RuntimeError(f"轨迹键重复：{key}")
            seen_keys.add(key)
            records.append(
                {
                    "identity": identity,
                    "states": payload["states"][index],
                    "forces": payload["forces"][index],
                    "times": payload["times"][index],
                    "cache_file": str(cache_file),
                }
            )
    return records


def split_cohort(records: list[dict], seed: int = 0) -> dict:
    count = len(records)
    if count not in COHORT_SPLITS:
        raise RuntimeError(
            f"当前同工况轨迹有 {count} 条。本阶段只接受 16 条（10/3/3）或 20 条（12/4/4）。"
        )
    sizes = COHORT_SPLITS[count]
    order = np.random.default_rng(seed).permutation(count)
    groups = {}
    start = 0
    for name, size in zip(("train", "valid", "test"), sizes):
        chosen = order[start:start + size]
        groups[name] = [records[int(index)]["identity"] for index in chosen]
        start += size
    keys = [item["trajectory_key"] for group in groups.values() for item in group]
    if len(keys) != len(set(keys)):
        raise RuntimeError("划分后出现重复轨迹。")
    return {
        "seed": seed,
        "count": count,
        "sizes": {"train": sizes[0], "valid": sizes[1], "test": sizes[2]},
        "rule": f"{count} 条完整轨迹，train/valid/test={sizes[0]}/{sizes[1]}/{sizes[2]}，seed={seed}。",
        "train": groups["train"],
        "valid": groups["valid"],
        "test": groups["test"],
    }
