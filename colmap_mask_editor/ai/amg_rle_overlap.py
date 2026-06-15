"""
V0.10: 2 つの SAM 2 uncompressed RLE の重なり量を dense 復号せずに計算する。

候補グループ化 (重複候補の削減) では、フル解像度マスクを毎回復号すると 8K で
非常に重い。SAM の uncompressed RLE は同じ Fortran order・同じ画像サイズなので、
前景 run の区間 (interval) 同士を線形に走査するだけで共通前景長を求められる。

提供する計算 (いずれも counts -> 整数 / 比率。dense マスクを作らない):
  rle_intersection_area : 前景 ∩ 前景 の画素数
  rle_union_area        : 前景 ∪ 前景 の画素数
  rle_iou               : intersection / union (union=0 なら 0.0)
  rle_containment       : inner ∩ outer / area(inner)  (inner が outer にどれだけ含まれるか)

numpy のみに依存する (torch / sam2 / PySide6 非依存)。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "foreground_intervals",
    "interval_overlap_length",
    "rle_intersection_area",
    "rle_union_area",
    "rle_iou",
    "rle_containment",
    "rle_area",
]


def rle_area(counts) -> int:
    """前景 run (奇数 index) の総和。amg_rle.rle_area と同義 (重複参照を避けるため再掲)。"""
    arr = np.asarray(counts)
    return int(arr[1::2].sum())


def foreground_intervals(counts) -> tuple[np.ndarray, np.ndarray]:
    """
    uncompressed RLE counts から前景 run の半開区間 [start, end) を返す。

    counts は「背景長, 前景長, 背景長, ...」の交互配列 (先頭は背景)。
    run i は flat 位置 [offs[i], offs[i+1]) を占め、i が奇数のとき前景。
    返り値: (starts, ends) いずれも int64 の 1 次元配列 (run 数ぶん)。
    """
    arr = np.asarray(counts, dtype=np.int64)
    if arr.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    offs = np.empty(arr.size + 1, dtype=np.int64)
    offs[0] = 0
    np.cumsum(arr, out=offs[1:])
    # 前景 run = 元 index 1,3,5,... -> offs の index 1,3,... が start, 2,4,... が end
    starts = offs[1::2]
    ends = offs[2::2]
    m = min(starts.size, ends.size)
    return starts[:m], ends[:m]


def interval_overlap_length(
    starts_a: np.ndarray, ends_a: np.ndarray,
    starts_b: np.ndarray, ends_b: np.ndarray,
) -> int:
    """
    昇順・非重複の 2 区間集合の共通長を 2 ポインタで合計する。O(na + nb)。
    """
    sa = starts_a.tolist(); ea = ends_a.tolist()
    sb = starts_b.tolist(); eb = ends_b.tolist()
    i = j = 0
    na = len(sa); nb = len(sb)
    total = 0
    while i < na and j < nb:
        lo = sa[i] if sa[i] > sb[j] else sb[j]
        hi = ea[i] if ea[i] < eb[j] else eb[j]
        if hi > lo:
            total += hi - lo
        if ea[i] < eb[j]:
            i += 1
        else:
            j += 1
    return int(total)


def rle_intersection_area(counts_a, counts_b) -> int:
    """前景 ∩ 前景 の画素数 (dense 復号なし)。"""
    sa, ea = foreground_intervals(counts_a)
    sb, eb = foreground_intervals(counts_b)
    if sa.size == 0 or sb.size == 0:
        return 0
    return interval_overlap_length(sa, ea, sb, eb)


def rle_union_area(counts_a, counts_b) -> int:
    """前景 ∪ 前景 の画素数 = area_a + area_b - intersection。"""
    area_a = rle_area(counts_a)
    area_b = rle_area(counts_b)
    inter = rle_intersection_area(counts_a, counts_b)
    return int(area_a + area_b - inter)


def rle_iou(counts_a, counts_b) -> float:
    """IoU = intersection / union。union が 0 (両方空) なら 0.0。"""
    inter = rle_intersection_area(counts_a, counts_b)
    area_a = rle_area(counts_a)
    area_b = rle_area(counts_b)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def rle_containment(inner_counts, outer_counts) -> float:
    """
    inner が outer にどれだけ含まれるか = (inner ∩ outer) / area(inner)。

    area(inner) が 0 なら 0.0。1.0 に近いほど inner は outer に包含される。
    """
    area_inner = rle_area(inner_counts)
    if area_inner <= 0:
        return 0.0
    inter = rle_intersection_area(inner_counts, outer_counts)
    return float(inter) / float(area_inner)
