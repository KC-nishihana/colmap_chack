"""
V0.8: SAM 2.1 Automatic Mask Generator ベンチマーク (実機・実モデルが必要)。

FHD/4K/8K のダミー画像で、高速/標準プリセットの全画像自動分割を計測する。
モデルは 1 回だけロードし、画像を 1 枚ずつ処理する (全画像同時GPUロードはしない)。

計測:
  モデルロード時間 / 画像読込時間 / generate時間 / RLE変換時間 / NPZ圧縮時間 /
  manifest保存時間 / 1枚あたり候補数 / NPZサイズ / JSONサイズ /
  dense換算サイズ / 圧縮率 / 最大VRAM / 最大RAM / OOM再試行回数 / 全体処理時間

出力:
  logs/sam2_amg_benchmark.json
  logs/sam2_amg_benchmark.csv

使い方:
    python colmap_mask_editor/tools/benchmark_sam2_amg.py ^
        --checkpoint models/sam2/sam2.1_hiera_small.pt --model sam2.1_hiera_small
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from ai import amg_manifest, amg_npz, amg_rle  # noqa: E402
from core.version import SAM2_COMMIT_SHA  # noqa: E402

# プラン (サイズ, 枚数, プリセット)
PLANS = [
    ("FHD", (1920, 1080), 10, "fast"),
    ("FHD", (1920, 1080), 10, "standard"),
    ("4K", (3840, 2160), 5, "fast"),
    ("4K", (3840, 2160), 5, "standard"),
    ("8K", (7680, 4320), 3, "fast"),
    ("8K", (7680, 4320), 3, "standard"),
]


def _make_image(w: int, h: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 30, np.uint8)
    for _ in range(40):
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(min(w, h) // 40, min(w, h) // 8))
        color = tuple(int(c) for c in rng.integers(60, 255, size=3))
        cv2.circle(img, (cx, cy), r, color, -1)
    for _ in range(20):
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
        x1, y1 = min(w, x0 + int(rng.integers(20, w // 6))), min(h, y0 + int(rng.integers(20, h // 6)))
        color = tuple(int(c) for c in rng.integers(60, 255, size=3))
        cv2.rectangle(img, (x0, y0), (x1, y1), color, -1)
    return img


def _peak_ram_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model", default="sam2.1_hiera_small")
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import torch  # noqa: WPS433
    from sam_backend.sam2_model_manager import Sam2ModelManager
    from sam_backend.sam2_amg_manager import Sam2AmgManager
    from ai import model_registry

    info = model_registry.get_model(args.model)

    mm = Sam2ModelManager()
    t0 = time.time()
    mm.load(model_id=args.model, config_name=info.config_name,
            checkpoint_path=args.checkpoint, precision=args.precision, device=args.device)
    model_load_sec = time.time() - t0
    amg = Sam2AmgManager(mm)

    rows: list[dict] = []
    out_dir = _PKG_ROOT / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_amg_bench_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    model = {"model_id": args.model, "sam2_commit": SAM2_COMMIT_SHA, "checkpoint_fingerprint": ""}

    for label, (w, h), count, preset in PLANS:
        settings = amg_manifest.preset_settings(preset)
        per_image = []
        for i in range(count):
            img = _make_image(w, h, seed=hash((label, preset, i)) & 0xFFFF)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            torch.cuda.reset_peak_memory_stats()
            t = time.time(); res = amg.generate(rgb, settings); gen_sec = time.time() - t
            anns = res.annotations

            t = time.time(); arrays = amg_npz.build_segment_arrays(anns, h, w); rle_sec = time.time() - t

            npz_path = tmp_dir / f"{label}_{preset}_{i}.npz"
            t = time.time(); sha = amg_npz.save_segments_npz(npz_path, arrays); npz_sec = time.time() - t

            man = amg_manifest.build_image_manifest(
                image_key=f"{label}/{i}.png", source_path=str(tmp_dir / f"{i}.png"),
                width=w, height=h, model=model, generator=settings, preset=preset,
                segment_count=int(arrays["segment_ids"].shape[0]),
                segment_ids=arrays["segment_ids"].tolist(), segments_npz_sha256=sha,
                processing_time_sec=gen_sec, fingerprint={"file_size": 0, "mtime_ns": 0},
            )
            man_path = tmp_dir / f"{label}_{preset}_{i}.json"
            t = time.time(); amg_manifest.atomic_write_json(man_path, man); man_sec = time.time() - t

            n = int(arrays["segment_ids"].shape[0])
            npz_size = npz_path.stat().st_size
            json_size = man_path.stat().st_size
            dense_size = n * h * w  # (N,H,W) uint8 換算
            per_image.append({
                "generate_sec": gen_sec, "rle_sec": rle_sec, "npz_sec": npz_sec,
                "manifest_sec": man_sec, "segment_count": n,
                "npz_bytes": npz_size, "json_bytes": json_size,
                "dense_bytes": dense_size,
                "compression_ratio": (dense_size / npz_size) if npz_size else 0,
                "peak_vram_mb": res.peak_vram_mb, "oom_retries": res.oom_retries,
                "peak_ram_mb": _peak_ram_mb(),
            })
            npz_path.unlink(missing_ok=True); man_path.unlink(missing_ok=True)
            amg.reclaim()

        def avg(key):
            return sum(d[key] for d in per_image) / len(per_image) if per_image else 0

        row = {
            "plan": f"{label} x{count} {preset}",
            "size": f"{w}x{h}", "count": count, "preset": preset,
            "model_load_sec": round(model_load_sec, 3),
            "avg_generate_sec": round(avg("generate_sec"), 3),
            "avg_rle_sec": round(avg("rle_sec"), 4),
            "avg_npz_sec": round(avg("npz_sec"), 4),
            "avg_manifest_sec": round(avg("manifest_sec"), 4),
            "avg_segment_count": round(avg("segment_count"), 1),
            "avg_npz_bytes": int(avg("npz_bytes")),
            "avg_json_bytes": int(avg("json_bytes")),
            "avg_dense_bytes": int(avg("dense_bytes")),
            "avg_compression_ratio": round(avg("compression_ratio"), 1),
            "max_vram_mb": max((d["peak_vram_mb"] for d in per_image), default=0),
            "max_ram_mb": round(max((d["peak_ram_mb"] for d in per_image), default=0), 1),
            "total_oom_retries": sum(d["oom_retries"] for d in per_image),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    mm.unload()

    (out_dir / "sam2_amg_benchmark.json").write_text(
        json.dumps({"model": args.model, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if rows:
        with open(out_dir / "sam2_amg_benchmark.csv", "w", newline="", encoding="utf-8") as f:
            wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wtr.writeheader()
            wtr.writerows(rows)
    print(f"\n保存: {out_dir/'sam2_amg_benchmark.json'} / .csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
