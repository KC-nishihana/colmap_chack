"""
V0.7: SAM 2.1 Video Predictor 伝播のベンチマーク (実機・要GPU/SAM2/チェックポイント)。

合成画像シーケンスを生成し、ステージング〜Video Predictorロード〜init_state〜
基準マスク追加〜前後伝播〜PNG保存〜VRAM/RAM を計測する。

使い方:
    python colmap_mask_editor/tools/benchmark_sam2_propagation.py ^
        --checkpoint models/sam2/sam2.1_hiera_small.pt

結果: logs/sam2_propagation_benchmark.json / .csv
8K がメモリ不足になる場合もエラーを隠さず記録する。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import tempfile
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

import cv2
import numpy as np

from ai import model_registry

SIZES = {"FHD": (1920, 1080), "4K": (3840, 2160), "8K": (7680, 4320)}


def _stage(frames_dir: Path, w: int, h: int, n: int) -> list:
    frames_dir.mkdir(parents=True, exist_ok=True)
    xs = []
    bw = max(60, w // 12)
    for i in range(n):
        img = np.zeros((h, w, 3), np.uint8)
        x = w // 4 + i * (w // (n * 3))
        cv2.rectangle(img, (x, h // 3), (x + bw, 2 * h // 3), (220, 220, 220), -1)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        buf.tofile(str(frames_dir / f"{i:06d}.jpg"))
        xs.append((x, bw))
    return xs


def _bench_case(label, w, h, n, ckpt, model_id, device, out_root) -> dict:
    import torch
    from sam_backend.sam2_video_manager import Sam2VideoManager
    from ai.propagation_staging import write_mask_png_atomic

    rec = {"label": label, "width": w, "height": h, "frames": n}
    job_dir = out_root / label
    frames_dir = job_dir / "frames"
    results_dir = job_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    info = model_registry.get_model(model_id)
    try:
        assert torch.cuda.is_available(), "CUDA が利用できません"
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        xs = _stage(frames_dir, w, h, n)
        rec["staging_sec"] = round(time.perf_counter() - t, 3)

        vm = Sam2VideoManager()
        t = time.perf_counter()
        vm.build(info.config_name, str(ckpt), device, "bf16")
        rec["model_load_sec"] = round(time.perf_counter() - t, 3)

        t = time.perf_counter()
        state = vm.init_state(str(frames_dir), offload_video_to_cpu=True)
        rec["init_state_sec"] = round(time.perf_counter() - t, 3)

        ref = n // 2
        x, bw = xs[ref]
        ref_mask = np.zeros((h, w), np.uint8)
        ref_mask[h // 3:2 * h // 3, x:x + bw] = 255
        t = time.perf_counter()
        vm.add_reference_mask(state, ref, 1, ref_mask > 0)
        rec["add_mask_sec"] = round(time.perf_counter() - t, 3)

        per_frame = []
        t = time.perf_counter()
        written = set()
        for f_idx, mask in vm.propagate(state, ref, n, reverse=False):
            ts = time.perf_counter()
            write_mask_png_atomic(results_dir / f"{f_idx:06d}.png", mask)
            per_frame.append(time.perf_counter() - ts)
            written.add(f_idx)
        rec["forward_sec"] = round(time.perf_counter() - t, 3)
        t = time.perf_counter()
        for f_idx, mask in vm.propagate(state, ref, n, reverse=True):
            if f_idx in written:
                continue
            write_mask_png_atomic(results_dir / f"{f_idx:06d}.png", mask)
            written.add(f_idx)
        rec["backward_sec"] = round(time.perf_counter() - t, 3)

        rec["frames_written"] = len(written)
        rec["avg_png_save_sec"] = round(float(np.mean(per_frame)), 4) if per_frame else 0.0
        rec["peak_vram_mb"] = int(torch.cuda.max_memory_allocated() / (1024 * 1024))
        t = time.perf_counter()
        vm.release()
        rec["release_sec"] = round(time.perf_counter() - t, 3)
        rec["success"] = True
    except Exception as e:  # noqa: BLE001 — 8K OOM 等を隠さず記録
        import traceback
        rec["success"] = False
        rec["error"] = repr(e)
        rec["traceback"] = traceback.format_exc()
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model", default=model_registry.DEFAULT_MODEL_ID)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--sizes", default="FHD,4K,8K")
    args = ap.parse_args()

    out_root = Path(tempfile.mkdtemp(prefix="prop_bench_"))
    cases = []
    for label in args.sizes.split(","):
        label = label.strip()
        if label not in SIZES:
            continue
        w, h = SIZES[label]
        cases.append(_bench_case(label, w, h, args.frames, args.checkpoint,
                                 args.model, args.device, out_root))
    # FHD 50枚も計測
    cases.append(_bench_case("FHD_50", 1920, 1080, 50, args.checkpoint,
                             args.model, args.device, out_root))

    logs = _PKG_ROOT.parent / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "sam2_propagation_benchmark.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    keys = sorted({k for c in cases for k in c.keys()})
    with open(logs / "sam2_propagation_benchmark.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for c in cases:
            w.writerow(c)
    print(json.dumps(cases, ensure_ascii=False, indent=2))
    return 0 if all(c.get("success") for c in cases) else 1


if __name__ == "__main__":
    sys.exit(main())
