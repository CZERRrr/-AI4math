"""检查并缓存同一 NU=0.01 工况的多个 NS 文件，然后固定轨迹划分。

不同 seed 只增加独立样本，不定义新工况。
原始 HDF5 不会在这里删除。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

if os.environ.get("OMP_NUM_THREADS") == "0":
    os.environ["OMP_NUM_THREADS"] = "1"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ns_experiment.data import (  # noqa: E402
    COHORT_SPLITS,
    SAME_CONDITION,
    build_file_cache,
    cache_path_for,
    inspect_source,
    load_cached_trajectories,
    split_cohort,
)

CACHE_DIR = ROOT / "data" / "cache" / "nu0.01_512_down8"
FRAMES = 201
DOWNSAMPLE = 8


def discover_paths(explicit: list[str] | None) -> list[Path]:
    if explicit:
        paths = [Path(item).expanduser().resolve() for item in explicit]
    else:
        paths = sorted(
            path.resolve()
            for path in (ROOT / "data").glob("ns_incom_inhom_2d_512-*.h5")
            if path.is_file() and not path.name.endswith(".downloading")
        )
    unique = []
    seen = set()
    for path in paths:
        key = path.resolve()
        if key in seen:
            raise RuntimeError(f"重复传入同一个文件：{path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        seen.add(key)
        unique.append(path)
    return unique


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", nargs="*", default=None, help="显式 HDF5 路径。缺省时使用 data/ 下已完成的 512 文件。")
    args = parser.parse_args()
    paths = discover_paths(args.paths)
    report_dir = ROOT / "results" / datetime.now().strftime("cohort_prepare_%Y%m%d_%H%M%S")
    report_dir.mkdir(parents=True)
    print("检查文件：", [str(path) for path in paths], flush=True)
    print(f"数据盘剩余：{shutil.disk_usage(ROOT / 'data').free} 字节", flush=True)
    inspections = [inspect_source(path) for path in paths]
    accepted = []
    rejected = []
    for item in inspections:
        if item.get("complete") and item.get("same_condition"):
            accepted.append(item)
        else:
            rejected.append(item)
            print("不并入：", item["name"], item.get("error") or item.get("condition_differences"), flush=True)
    cache_files = []
    for item in accepted:
        source = Path(item["path"])
        destination = cache_path_for(CACHE_DIR, source, FRAMES, DOWNSAMPLE)
        if destination.is_file():
            print("已有缓存：", destination, flush=True)
        else:
            build_file_cache(source, CACHE_DIR, FRAMES, DOWNSAMPLE)
        cache_files.append(destination)
    records = load_cached_trajectories(cache_files) if cache_files else []
    split = None
    split_error = None
    try:
        if records:
            split = split_cohort(records, seed=0)
    except RuntimeError as exc:
        split_error = str(exc)
    manifest = {
        "reference_condition": SAME_CONDITION,
        "note": "seed 可以不同。不同 seed 不是新工况。",
        "frames_cached": FRAMES,
        "downsample": DOWNSAMPLE,
        "train_frames_later": 101,
        "disk_free_bytes": shutil.disk_usage(ROOT / "data").free,
        "inspected": inspections,
        "rejected": [item["name"] for item in rejected],
        "accepted_files": [item["name"] for item in accepted],
        "trajectory_count": len(records),
        "cache_files": [
            {"path": str(path), "bytes": path.stat().st_size}
            for path in cache_files
        ],
        "cache_bytes": sum(path.stat().st_size for path in cache_files),
        "split_rule_if_count": {str(count): list(sizes) for count, sizes in COHORT_SPLITS.items()},
        "split_error": split_error,
        "target": "16 或 20 条同工况轨迹。20 条需要再下一个约 9.9GB 文件，当前余量不够时不强行下载。",
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    write_json(CACHE_DIR / "manifest.json", manifest)
    write_json(report_dir / "manifest.json", manifest)
    if split is not None:
        write_json(CACHE_DIR / "split_ids.json", split)
        write_json(report_dir / "split_ids.json", split)
    lines = [
        "# 同工况 NS 数据准备",
        "",
        f"- 检查到的完整且同工况文件：{len(accepted)}",
        f"- 独立轨迹数：{len(records)}",
        f"- 缓存总字节：{manifest['cache_bytes']}",
        f"- 缓存目录：`{CACHE_DIR}`",
        f"- 划分：{split['rule'] if split else split_error or '轨迹数还不够'}",
        "",
        "原始 HDF5 没有删除。",
        "",
    ]
    if split is None:
        lines.append(
            f"距离 16 条还差 {max(0, 16 - len(records))} 条，距离 20 条还差 {max(0, 20 - len(records))} 条。"
            "不能复制已有轨迹来补数量。"
        )
    (report_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print("准备记录：", report_dir, flush=True)
    return 0 if split is not None else 2


if __name__ == "__main__":
    sys.exit(main())
