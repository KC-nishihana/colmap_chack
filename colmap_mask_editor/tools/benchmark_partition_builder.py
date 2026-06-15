"""
V0.9: 完全被覆 partition 生成のベンチマーク (CPU 専用・torch/sam2 非依存)。

FHD / 4K / 8K の合成画像に対し、粗い / 標準プリセットで partition を生成し、各
ステージ時間・葉/ノード数・partition.npz サイズ・coverage・クリック判定時間・
最終マスク生成時間を計測して logs/partition_benchmark.{json,csv} へ保存する。

実行例:
  python -m tools.benchmark_partition_builder --sizes fhd,4k --presets coarse,standard
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import tracemalloc
from pathlib import Path

import cv2
import numpy as np

from ai import partition_npz, partition_manifest as pman
from ai import partition_mask_composer as pmc
from ai.partition_hit_test import PartitionHitTester
from ai.partition_tree import cut_tree_to_count, PartitionTree
from partition_backend import partition_builder as builder

SIZES = {
    "fhd": (1920, 1080),
    "4k": (3840, 2160),
    "8k": (7680, 4320),
}
PRESETS = {
    "coarse": {"base_region_count": 800, "default_visible_count": 30,
               "min_region_area_ratio": 10},
    "standard": {"base_region_count": 1500, "default_visible_count": 70,
                 "min_region_area_ratio": 5},
}


def _make_image(w: int, h: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tile = rng.integers(0, 256, size=(16, 16, 3)).astype(np.uint8)
    ry = h // 16 + 1
    rx = w // 16 + 1
    img = np.kron(tile, np.ones((ry, rx, 1), dtype=np.uint8))[:h, :w]
    noise = rng.integers(-12, 13, size=(h, w, 3))
    return np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)


def run_case(size_key: str, preset_key: str, workdir: Path) -> dict:
    w, h = SIZES[size_key]
    img = _make_image(w, h, seed=hash((w, h)) & 0xFFFF)
    img_path = workdir / f"{size_key}.png"
    img_path.write_bytes(cv2.imencode(".png", img)[1].tobytes())
    out = workdir / f"cache_{size_key}_{preset_key}"

    settings = {"backend": "auto", "working_max_side": 2048}
    settings.update(PRESETS[preset_key])

    stage_times: dict[str, float] = {}
    last = {"t": time.time(), "stage": None}

    def progress(stage, frac, info):
        now = time.time()
        if last["stage"] is not None:
            stage_times[last["stage"]] = stage_times.get(last["stage"], 0.0) + (now - last["t"])
        last["t"] = now
        last["stage"] = stage

    tracemalloc.start()
    t0 = time.time()
    manifest = builder.build_partition(
        img_path, image_key=size_key, output_dir=out, settings=settings,
        progress=progress)
    total = time.time() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    data = partition_npz.load_partition_npz(out / pman.PARTITION_NPZ_NAME)
    npz_size = (out / pman.PARTITION_NPZ_NAME).stat().st_size
    leaf_count = int(data["leaf_count"][0])
    node_count = int(data["node_count"][0])

    # 30 リージョンへ cut する時間
    tree = PartitionTree.from_npz(data)
    tc = time.time()
    visible = cut_tree_to_count(tree, 30)
    cut_time = time.time() - tc

    # クリック判定時間 (100 回平均)
    ht = PartitionHitTester(data)
    rng = np.random.default_rng(1)
    pts = rng.integers(0, [w, h], size=(100, 2))
    tck = time.time()
    for x, y in pts:
        ht.leaf_at(int(x), int(y))
    click_time = (time.time() - tck) / 100.0

    # 最終マスク生成時間 (全 keep)
    parent = data["node_parent"]
    lut = pmc.leaf_decision_values(parent, leaf_count, {}, unreviewed_as="keep")
    tm = time.time()
    mask = pmc.compose_mask(data["run_region_ids"], data["run_lengths"], h, w, lut)
    mask_time = time.time() - tm
    assert mask.shape == (h, w)

    areas = np.asarray(data["node_area"])[:leaf_count]
    cov = manifest["coverage"]
    return {
        "size": size_key, "preset": preset_key,
        "width": w, "height": h,
        "backend_used": manifest["backend_used"],
        "working_width": manifest["working_width"],
        "working_height": manifest["working_height"],
        "leaf_count": leaf_count, "node_count": node_count,
        "root_id": int(data["root_id"][0]),
        "default_visible_count": manifest["default_visible_count"],
        "coverage_ratio": cov["coverage_ratio"],
        "unassigned_pixels": cov["unassigned_pixels"],
        "overlap_pixels": cov["overlap_pixels"],
        "min_area": int(areas.min()), "median_area": int(np.median(areas)),
        "max_area": int(areas.max()),
        "partition_npz_bytes": int(npz_size),
        "total_sec": round(total, 3),
        "stage_sec": {k: round(v, 3) for k, v in stage_times.items()},
        "peak_python_mem_mb": round(peak / 1e6, 1),
        "cut30_sec": round(cut_time, 4),
        "click_sec": round(click_time, 6),
        "final_mask_sec": round(mask_time, 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="fhd,4k,8k")
    ap.add_argument("--presets", default="coarse,standard")
    ap.add_argument("--out", default="logs")
    args = ap.parse_args()

    sizes = [s.strip() for s in args.sizes.split(",") if s.strip() in SIZES]
    presets = [p.strip() for p in args.presets.split(",") if p.strip() in PRESETS]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    workdir = out_dir / "_bench_work"
    workdir.mkdir(parents=True, exist_ok=True)

    results = []
    for sk in sizes:
        for pk in presets:
            print(f"[bench] {sk} / {pk} ...", flush=True)
            try:
                r = run_case(sk, pk, workdir)
                print(f"  total={r['total_sec']}s leaves={r['leaf_count']} "
                      f"cov={r['coverage_ratio']} npz={r['partition_npz_bytes']}B", flush=True)
                results.append(r)
            except Exception as e:  # noqa: BLE001
                print(f"  FAILED: {e}", flush=True)
                results.append({"size": sk, "preset": pk, "error": str(e)})

    (out_dir / "partition_benchmark.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    # CSV (stage_sec は除く)
    flat_keys = [k for k in results[0].keys() if k != "stage_sec"] if results else []
    with open(out_dir / "partition_benchmark.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(flat_keys)
        for r in results:
            wr.writerow([r.get(k, "") for k in flat_keys])
    print(f"[bench] 保存: {out_dir/'partition_benchmark.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
