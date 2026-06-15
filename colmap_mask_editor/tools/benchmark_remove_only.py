"""
V0.10: REMOVE_ONLY の効果測定 / 4K・8K 操作時間ベンチマーク。

実際の segments.npz パイプライン (amg_npz.build_segment_arrays / save_segments_npz) で
NPZ を生成し、以下を計測する:
  - review_index 構築時間 (重複グループ計算 + 確認順スコア)
  - 候補表示数: 総数 vs 代表候補数 (従来比の削減率)
  - 最終マスク生成時間 (compose_final_mask MODE_EXCLUDE_REMOVE 再利用)
  - 1 判断あたりの更新時間 (REMOVE 和集合 + 画素率の再計算)

GUI / torch / sam2 は使用しない (numpy + amg_* のみ)。

実行:
  conda run -p <env> python colmap_mask_editor/tools/benchmark_remove_only.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai import amg_manifest, amg_npz, amg_rle  # noqa: E402
from ai import amg_remove_only as ro  # noqa: E402
from ai.amg_hit_test import MaskDecodeCache  # noqa: E402
from core.amg_review_index_worker import ensure_review_index  # noqa: E402


def _rect_counts(h: int, w: int, y0: int, y1: int, x0: int, x1: int) -> list[int]:
    """矩形 [y0:y1, x0:x1) の uncompressed RLE counts を dense を作らず生成する。"""
    # Fortran order: 列ごとに高さ h。前景は列 x0..x1-1 の行 y0..y1-1。
    counts: list[int] = []
    pos = 0  # 直前 run の parity (False=bg)
    cur_bg = 0
    runs: list[int] = []
    # 列を走査
    for x in range(w):
        if x0 <= x < x1:
            # 上 bg (y0), 前景 (y1-y0), 下 bg (h-y1)
            top = y0
            fg = y1 - y0
            bot = h - y1
            # bg 連結
            cur_bg += top
            runs_append_bg_fg(runs, cur_bg, fg)
            cur_bg = bot
        else:
            cur_bg += h
    # 末尾 bg
    counts = runs
    counts.append(cur_bg)
    # 先頭は bg から開始する必要がある。runs_append_bg_fg は [bg, fg, bg, fg...] で積む
    return counts


def runs_append_bg_fg(runs: list[int], bg: int, fg: int) -> None:
    runs.append(bg)
    runs.append(fg)


def _make_annotations(h: int, w: int, n_objects: int, dup_per: int, rng):
    """n_objects 個の矩形 + それぞれ dup_per 個の近似重複候補を作る。"""
    anns = []
    for _ in range(n_objects):
        bw = int(rng.integers(w // 20, w // 6))
        bh = int(rng.integers(h // 20, h // 6))
        x0 = int(rng.integers(0, w - bw))
        y0 = int(rng.integers(0, h - bh))
        base = (y0, y0 + bh, x0, x0 + bw)
        variants = [base]
        for _ in range(dup_per):
            dx = int(rng.integers(-2, 3)); dy = int(rng.integers(-2, 3))
            yy0 = max(0, min(h - 1, base[0] + dy)); yy1 = max(yy0 + 1, min(h, base[1] + dy))
            xx0 = max(0, min(w - 1, base[2] + dx)); xx1 = max(xx0 + 1, min(w, base[3] + dx))
            variants.append((yy0, yy1, xx0, xx1))
        for (vy0, vy1, vx0, vx1) in variants:
            counts = _rect_counts(h, w, vy0, vy1, vx0, vx1)
            area = amg_rle.rle_area(counts)
            anns.append({
                "segmentation": {"size": [h, w], "counts": counts},
                "area": int(area),
                "bbox": [vx0, vy0, vx1 - vx0, vy1 - vy0],
                "predicted_iou": float(rng.uniform(0.7, 0.99)),
                "stability_score": float(rng.uniform(0.8, 0.99)),
                "point_coords": [[float((vx0 + vx1) / 2), float((vy0 + vy1) / 2)]],
                "crop_box": [0, 0, w, h],
            })
    return anns


def bench(label: str, h: int, w: int, n_objects: int, dup_per: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    print(f"\n=== {label}  ({w}x{h}, objects={n_objects}, dup_per={dup_per}) ===")
    t0 = time.perf_counter()
    anns = _make_annotations(h, w, n_objects, dup_per, rng)
    arrays = amg_npz.build_segment_arrays(anns, h, w)
    total = int(arrays["segment_ids"].shape[0])
    print(f"候補総数: {total}  (NPZ 構築 {time.perf_counter() - t0:.2f}s)")

    with tempfile.TemporaryDirectory() as td:
        cache_dir = Path(td)
        amg_npz.save_segments_npz(cache_dir / amg_manifest.SEGMENTS_NPZ_NAME, arrays)

        t0 = time.perf_counter()
        result = ensure_review_index(cache_dir, iou_threshold=0.85, containment_threshold=0.95)
        t_index = time.perf_counter() - t0
        reps = result.group_count
        print(f"review_index 構築: {t_index:.2f}s  代表候補数: {reps}  "
              f"表示削減: {100 * (1 - reps / total):.1f}% 減")

        # 模擬レビュー: priority 順上位の代表を REMOVE していく
        idx_arrays = result.arrays
        seg_ids = idx_arrays["segment_ids"]
        reps_mask = idx_arrays["representative_segment_ids"] == seg_ids
        rep_indices = [int(i) for i in np.flatnonzero(reps_mask)]
        pri = idx_arrays["priority_scores"]
        rep_indices.sort(key=lambda i: -float(pri[i]))

        cache = MaskDecodeCache(arrays, max_size=16)
        base = np.ones((h, w), dtype=bool)
        decisions: dict[str, str] = {}
        n_clicks = min(8, len(rep_indices))
        per_decision = []
        for k in range(n_clicks):
            i = rep_indices[k]
            decisions[str(int(seg_ids[i]))] = "remove"
            t0 = time.perf_counter()
            rem = cache.union([j for j, s in enumerate(seg_ids.tolist())
                               if decisions.get(str(int(s))) == "remove"])
            st = ro.pixel_stats(base, rem)
            per_decision.append(time.perf_counter() - t0)
        if per_decision:
            print(f"1判断あたり更新 (REMOVE和集合+画素率): 平均 {1000 * np.mean(per_decision):.1f} ms "
                  f"(最終除外率 {st.excluded_ratio * 100:.1f}%)")

        # 最終マスク生成 (既存 compose_final_mask MODE_EXCLUDE_REMOVE 再利用)
        t0 = time.perf_counter()
        final = ro.compose_remove_only_final(arrays, decisions, base_mode=ro.BASE_FULL)
        t_final = time.perf_counter() - t0
        assert final.dtype == np.uint8 and set(np.unique(final).tolist()).issubset({0, 255})
        print(f"最終マスク生成: {t_final:.2f}s  (REMOVE {n_clicks}候補 -> 除外画素 "
              f"{int((final == 0).sum()):,})")
        print(f"クリック数(従来比): 全候補レビュー {total} 回 -> REMOVE_ONLY 約 {n_clicks} 回 "
              f"({100 * n_clicks / total:.1f}%)")


def main():
    bench("4K", 2160, 3840, n_objects=40, dup_per=2)
    bench("8K", 4320, 7680, n_objects=40, dup_per=2)


if __name__ == "__main__":
    main()
