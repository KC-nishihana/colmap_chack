"""
Phase 23: SAM 2 ベンチマーク (実機・実モデルが必要)。

FHD / 4K / 8K のダミー画像で、Worker起動・モデルロード・Embedding・推論・
VRAM・NPZ書込/読込・子プロセス終了時間を計測し、JSON/CSV へ保存する。

使い方:
    python colmap_mask_editor/tools/benchmark_sam2.py ^
        --checkpoint models/sam2/sam2.1_hiera_small.pt --model sam2.1_hiera_small

GUI と同じ QProcess 経路を使い、ai.ai_session.AiSession 経由で計測する。
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

from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer  # noqa: E402

from ai import model_registry  # noqa: E402
from ai.ai_session import AiSession, AiUiState  # noqa: E402

SIZES = {
    "FHD": (1920, 1080),
    "4K": (3840, 2160),
    "8K": (7680, 4320),
}


def _wait(app, predicate, timeout_ms=200000):
    loop = QEventLoop()
    t = QTimer()
    t.setSingleShot(True)
    t.timeout.connect(loop.quit)
    t.start(timeout_ms)
    while not predicate() and t.isActive():
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
    return predicate()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model", default=model_registry.DEFAULT_MODEL_ID)
    ap.add_argument("--precision", default="bf16")
    args = ap.parse_args()

    app = QCoreApplication.instance() or QCoreApplication([])

    tmpdir = _PKG_ROOT.parent / "logs" / "bench_images"
    tmpdir.mkdir(parents=True, exist_ok=True)

    results = []
    sess = AiSession()

    t0 = time.perf_counter()
    sess.start_worker()
    _wait(app, lambda: sess.state in (AiUiState.WORKER_READY, AiUiState.ERROR))
    worker_start = time.perf_counter() - t0
    if sess.state == AiUiState.ERROR:
        print("Worker起動に失敗しました")
        return 1

    t0 = time.perf_counter()
    sess.load_model(args.model, args.checkpoint, args.precision, "cuda:0")
    _wait(app, lambda: sess.state in (AiUiState.MODEL_READY, AiUiState.ERROR))
    model_load = time.perf_counter() - t0

    for name, (w, h) in SIZES.items():
        img = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
        path = tmpdir / f"bench_{name}.png"
        cv2.imwrite(str(path), img)

        row = {"size": name, "width": w, "height": h,
               "worker_start_sec": round(worker_start, 3),
               "model_load_sec": round(model_load, 3)}

        t0 = time.perf_counter()
        sess.set_image(str(path))
        _wait(app, lambda: sess.state in (AiUiState.PROMPT_EDITING, AiUiState.ERROR))
        row["set_image_sec"] = round(time.perf_counter() - t0, 3)

        # 1回目推論
        sess.prompts.add_point(w // 2, h // 2, positive=True)
        t0 = time.perf_counter()
        sess.predict()
        _wait(app, lambda: sess.state in (AiUiState.PREVIEW, AiUiState.ERROR, AiUiState.PROMPT_EDITING))
        row["predict1_sec"] = round(time.perf_counter() - t0, 3)
        if sess.result is not None:
            row["mask_count"] = sess.result.mask_count

        # 2回目推論 (点追加)
        sess.discard_preview()
        sess.prompts.add_point(w // 2 + 50, h // 2 + 50, positive=True)
        t0 = time.perf_counter()
        sess.predict()
        _wait(app, lambda: sess.state in (AiUiState.PREVIEW, AiUiState.ERROR, AiUiState.PROMPT_EDITING))
        row["predict2_sec"] = round(time.perf_counter() - t0, 3)

        row["vram_allocated_mb"] = sess.hello_info.get("vram_allocated_mb", 0)
        results.append(row)
        sess.invalidate_image()

    t0 = time.perf_counter()
    sess.shutdown()
    shutdown = time.perf_counter() - t0
    for r in results:
        r["shutdown_sec"] = round(shutdown, 3)

    logs = _PKG_ROOT.parent / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "sam2_benchmark.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    if results:
        with open(logs / "sam2_benchmark.csv", "w", newline="", encoding="utf-8-sig") as f:
            wri = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            wri.writeheader()
            wri.writerows(results)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\n保存: {logs / 'sam2_benchmark.json'} / .csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
