"""按官方清单下载指定的 NS 分片。

地址只来自 data/darus_ns512_catalog.json 的 url 字段。
空间不够就停止，不删除已经下完或下到一半的文件。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CATALOG = DATA / "darus_ns512_catalog.json"
DEFAULT_NAMES = [
    "ns_incom_inhom_2d_512-241.h5",
    "ns_incom_inhom_2d_512-194.h5",
    "ns_incom_inhom_2d_512-45.h5",
]
MARGIN = 2 * 1024**3


def main() -> int:
    names = sys.argv[1:] or DEFAULT_NAMES
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    by_name = {item["name"]: item for item in catalog["data"]["items"]}
    for name in names:
        if name not in by_name:
            raise SystemExit(f"官方清单里没有 {name}，不能猜测下载地址。")
        item = by_name[name]
        dest = DATA / name
        partial = DATA / f"{name}.downloading"
        size = int(item["size_in_bytes"])
        md5 = item["md5"]
        url = item["url"]
        print(f"PLAN {name} size={size} md5={md5} url={url}", flush=True)
        if dest.is_file() and dest.stat().st_size == size:
            print(f"SKIP existing complete {name}", flush=True)
            continue
        free = shutil.disk_usage(DATA).free
        already = partial.stat().st_size if partial.exists() else 0
        print(f"FREE {free}", flush=True)
        if free < size - already + MARGIN:
            print(f"STOP not enough space for {name}", flush=True)
            return 2
        cmd = [
            "curl", "-L", "--http1.1", "--fail",
            "--retry", "20", "--retry-all-errors", "--retry-delay", "5", "-C", "-",
            "-o", str(partial), url,
        ]
        # HTTP/2 被对端取消时 curl 返回 92，--retry 默认不覆盖这类错误。
        # 半成品留在 .downloading，外层再续传几次。
        completed = False
        for attempt in range(1, 6):
            print(f"RUN attempt={attempt}", " ".join(cmd), flush=True)
            result = subprocess.run(cmd)
            got = partial.stat().st_size if partial.exists() else 0
            print(f"ATTEMPT {attempt} exit={result.returncode} bytes={got}", flush=True)
            if result.returncode == 0 and got == size:
                completed = True
                break
        if not completed:
            print(f"DOWNLOAD INCOMPLETE {name}: {got} / {size}", flush=True)
            return 2
        print(f"DOWNLOADED {got}", flush=True)
        if got != size:
            print(f"SIZE MISMATCH {name}: {got} != {size}", flush=True)
            return 2
        digest = hashlib.md5()
        with partial.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        found = digest.hexdigest()
        print(f"MD5 {found}", flush=True)
        if found != md5:
            print(f"MD5 MISMATCH {name}", flush=True)
            return 2
        partial.rename(dest)
        print(f"OK {dest}", flush=True)
    print("DOWNLOAD_STAGE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
