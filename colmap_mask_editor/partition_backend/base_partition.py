"""
V0.9: 基礎リージョン分割の共有ユーティリティ (numpy + OpenCV のみ)。

torch / sam2 / PySide6 / scipy / scikit-image に依存しない。完全被覆 (全画素が
正の region_id を持ち、未所属 0・重複 0) を保証するための再ラベル・連結性確保・
小領域統合・座標変換・検証を提供する。

ラベル規約: region_id は 1..K の連続値 (0 = 未所属を残さない)。
作業解像度ラベルを元解像度へ戻すときは必ず最近傍補間を使う。
"""

from __future__ import annotations

from enum import Enum

import cv2
import numpy as np

__all__ = [
    "BasePartitionBackend",
    "run_base_partition",
    "compute_working_size",
    "to_lab",
    "gradient_magnitude",
    "relabel_sequential",
    "region_areas",
    "region_bboxes",
    "region_mean_lab",
    "neighbor_pairs",
    "enforce_connectivity",
    "merge_small_regions",
    "coverage_stats",
    "validate_base_labels",
    "upscale_labels_nearest",
    "BaseLabelError",
]


class BasePartitionBackend(Enum):
    AUTO = "auto"
    SLIC = "slic"
    GRID_WATERSHED = "grid_watershed"


class BaseLabelError(ValueError):
    """基礎ラベルが完全被覆要件を満たさないときに送出する。"""


def run_base_partition(
    image_bgr: np.ndarray,
    backend: "BasePartitionBackend | str",
    *,
    seed_spacing: int | None = None,
    base_region_count: int | None = None,
    slic_region_size: int | None = None,
    slic_ruler: float = 10.0,
    min_area: int = 0,
) -> tuple[np.ndarray, str]:
    """
    backend を選んで作業解像度ラベルを生成する。返り値 (labels, backend_used)。

    AUTO: SLIC が使えれば SLICO、無ければ GRID_WATERSHED へフォールバック。
    SLIC を明示指定して ximgproc が無い場合は SlicUnavailableError を伝播する。
    """
    from partition_backend import slic_backend, watershed_backend

    if isinstance(backend, BasePartitionBackend):
        backend = backend.value
    backend = str(backend)

    if backend == BasePartitionBackend.SLIC.value:
        labels = slic_backend.slic_superpixels(
            image_bgr, region_size=slic_region_size, ruler=slic_ruler,
            base_region_count=base_region_count, min_area=min_area,
        )
        return labels, BasePartitionBackend.SLIC.value
    if backend == BasePartitionBackend.GRID_WATERSHED.value:
        labels = watershed_backend.grid_watershed(
            image_bgr, seed_spacing=seed_spacing,
            base_region_count=base_region_count, min_area=min_area,
        )
        return labels, BasePartitionBackend.GRID_WATERSHED.value
    if backend == BasePartitionBackend.AUTO.value:
        if slic_backend.slic_available():
            labels = slic_backend.slic_superpixels(
                image_bgr, region_size=slic_region_size, ruler=slic_ruler,
                base_region_count=base_region_count, min_area=min_area,
            )
            return labels, BasePartitionBackend.SLIC.value
        labels = watershed_backend.grid_watershed(
            image_bgr, seed_spacing=seed_spacing,
            base_region_count=base_region_count, min_area=min_area,
        )
        return labels, BasePartitionBackend.GRID_WATERSHED.value
    raise ValueError(f"不明なバックエンド: {backend!r}")


# ------------------------------------------------------------------ #
# 解像度・色変換
# ------------------------------------------------------------------ #


def compute_working_size(orig_w: int, orig_h: int, working_max_side: int) -> tuple[int, int]:
    """
    長辺が working_max_side を超える場合の作業解像度 (ww, wh) を返す。

    working_max_side <= 0 は原寸。アスペクト比を保ち最近傍で縮小する前提。
    """
    ow, oh = int(orig_w), int(orig_h)
    ms = int(working_max_side)
    if ms <= 0 or max(ow, oh) <= ms:
        return ow, oh
    scale = ms / float(max(ow, oh))
    ww = max(1, int(round(ow * scale)))
    wh = max(1, int(round(oh * scale)))
    return ww, wh


def to_lab(image_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 画像を Lab float32 (L,a,b) へ変換する。"""
    img = np.asarray(image_bgr)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    return lab


def gradient_magnitude(lab: np.ndarray) -> np.ndarray:
    """Lab の L チャンネルから Sobel 勾配強度 (float32) を計算する。"""
    L = np.asarray(lab)[..., 0].astype(np.float32)
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


# ------------------------------------------------------------------ #
# ラベル基本操作
# ------------------------------------------------------------------ #


def relabel_sequential(labels: np.ndarray) -> np.ndarray:
    """
    任意の整数ラベルを 1..K の連続値へ決定的に再割当てする。

    出現順 (C-order の最初の出現位置) に 1 から振る。0 や負値も 1 以上へ。
    """
    arr = np.asarray(labels)
    flat = arr.reshape(-1)
    uniq, inverse = np.unique(flat, return_inverse=True)
    # uniq は昇順。新ラベル = その昇順 index + 1。決定的。
    new_flat = (inverse + 1).astype(np.int32)
    return new_flat.reshape(arr.shape)


def region_areas(labels: np.ndarray, k: int | None = None) -> np.ndarray:
    """region_id ごとの画素数 (index 0 は未使用)。labels は 1..K 前提。"""
    arr = np.asarray(labels).reshape(-1).astype(np.int64)
    if k is None:
        k = int(arr.max()) if arr.size else 0
    return np.bincount(arr, minlength=k + 1)


def region_bboxes(labels: np.ndarray, k: int | None = None) -> np.ndarray:
    """
    各 region_id の bbox [x, y, w, h] を返す (shape (K+1, 4), index 0 未使用)。
    存在しない id は [0,0,0,0]。
    """
    arr = np.asarray(labels)
    h, w = arr.shape
    if k is None:
        k = int(arr.max())
    ys, xs = np.indices((h, w))
    flat = arr.reshape(-1)
    xr = xs.reshape(-1)
    yr = ys.reshape(-1)
    minx = np.full(k + 1, w, dtype=np.int64)
    miny = np.full(k + 1, h, dtype=np.int64)
    maxx = np.full(k + 1, -1, dtype=np.int64)
    maxy = np.full(k + 1, -1, dtype=np.int64)
    np.minimum.at(minx, flat, xr)
    np.minimum.at(miny, flat, yr)
    np.maximum.at(maxx, flat, xr)
    np.maximum.at(maxy, flat, yr)
    bbox = np.zeros((k + 1, 4), dtype=np.int64)
    present = maxx >= 0
    bbox[present, 0] = minx[present]
    bbox[present, 1] = miny[present]
    bbox[present, 2] = maxx[present] - minx[present] + 1
    bbox[present, 3] = maxy[present] - miny[present] + 1
    return bbox


def region_mean_lab(labels: np.ndarray, lab: np.ndarray, k: int | None = None) -> np.ndarray:
    """各 region_id の平均 Lab (shape (K+1, 3))。"""
    arr = np.asarray(labels).reshape(-1).astype(np.int64)
    if k is None:
        k = int(arr.max())
    area = np.bincount(arr, minlength=k + 1).astype(np.float64)
    area[area == 0] = 1.0
    out = np.zeros((k + 1, 3), dtype=np.float64)
    lab_flat = np.asarray(lab).reshape(-1, 3).astype(np.float64)
    for c in range(3):
        s = np.bincount(arr, weights=lab_flat[:, c], minlength=k + 1)
        out[:, c] = s / area
    return out.astype(np.float32)


def neighbor_pairs(labels: np.ndarray) -> np.ndarray:
    """
    4 近傍で異なる region_id を持つ隣接ペア (a<b) の一意配列 (shape (M, 2))。
    斜め接触は含めない。
    """
    arr = np.asarray(labels)
    a_h = arr[:, :-1].reshape(-1)
    b_h = arr[:, 1:].reshape(-1)
    a_v = arr[:-1, :].reshape(-1)
    b_v = arr[1:, :].reshape(-1)
    a = np.concatenate([a_h, a_v]).astype(np.int64)
    b = np.concatenate([b_h, b_v]).astype(np.int64)
    diff = a != b
    a, b = a[diff], b[diff]
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    pairs = np.stack([lo, hi], axis=1)
    if pairs.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.unique(pairs, axis=0)


# ------------------------------------------------------------------ #
# 連結性確保 (disconnected な同一 id を分割)
# ------------------------------------------------------------------ #


def enforce_connectivity(labels: np.ndarray) -> np.ndarray:
    """
    各 region_id が 4 連結であることを保証する。非連結な id は別 id へ分割する。

    bbox に切り出して cv2.connectedComponents を使う (全体走査を避け、作業量を
    各領域の bbox 面積合計に抑える)。返り値は 1..K' の連続ラベル。
    """
    arr = relabel_sequential(labels)
    k = int(arr.max())
    bbox = region_bboxes(arr, k)
    out = arr.copy().astype(np.int32)
    next_id = k + 1
    for rid in range(1, k + 1):
        x, y, bw, bh = (int(v) for v in bbox[rid])
        if bw == 0:
            continue
        sub = arr[y:y + bh, x:x + bw]
        mask = (sub == rid).astype(np.uint8)
        ncomp, comp = cv2.connectedComponents(mask, connectivity=4)
        if ncomp <= 2:  # 0=背景 + 1 前景成分 のみ -> 連結
            continue
        # 最大成分は rid を維持、残りは新 id
        sizes = np.bincount(comp.reshape(-1))
        sizes[0] = 0  # 背景除外
        keep = int(np.argmax(sizes))
        sub_out = out[y:y + bh, x:x + bw]
        for c in range(1, ncomp):
            if c == keep:
                continue
            sub_out[comp == c] = next_id
            next_id += 1
    return relabel_sequential(out)


# ------------------------------------------------------------------ #
# 小領域統合
# ------------------------------------------------------------------ #


def merge_small_regions(
    labels: np.ndarray,
    lab: np.ndarray,
    min_area: int,
    *,
    max_passes: int = 64,
) -> np.ndarray:
    """
    面積 < min_area の領域を、最も色 (平均 Lab) が近い隣接領域へ統合する。

    隣接同士のみ統合するため連結性は保たれる。決定的 (面積昇順・id 昇順)。
    """
    arr = relabel_sequential(labels).astype(np.int32)
    min_area = int(min_area)
    if min_area <= 1:
        return arr

    for _ in range(max_passes):
        k = int(arr.max())
        if k <= 1:
            break
        areas = region_areas(arr, k)
        small = [rid for rid in range(1, k + 1) if 0 < areas[rid] < min_area]
        if not small:
            break
        pairs = neighbor_pairs(arr)
        if pairs.shape[0] == 0:
            break
        adj: dict[int, set[int]] = {}
        for a, b in pairs:
            adj.setdefault(int(a), set()).add(int(b))
            adj.setdefault(int(b), set()).add(int(a))
        means = region_mean_lab(arr, lab, k).astype(np.float64)

        # 小領域を面積昇順 (同面積は id 昇順) で処理
        small.sort(key=lambda r: (int(areas[r]), r))
        mapping: dict[int, int] = {}
        for s in small:
            neighbors = adj.get(s, set())
            if not neighbors:
                continue
            # 非小領域を優先 (吸収先が安定)。なければ全隣接から選ぶ。
            cand = [t for t in neighbors if areas[t] >= min_area] or list(neighbors)
            ms = means[s]
            t = min(cand, key=lambda t: (float(np.linalg.norm(means[t] - ms)), t))
            mapping[s] = t
        if not mapping:
            break
        # 連鎖 (s->t->u) を解決して終端へ
        def resolve(r: int, seen: set[int]) -> int:
            while r in mapping:
                if r in seen:
                    break
                seen.add(r)
                r = mapping[r]
            return r
        lut = np.arange(k + 1, dtype=np.int32)
        for s in mapping:
            lut[s] = resolve(s, set())
        arr = relabel_sequential(lut[arr])
    return arr


# ------------------------------------------------------------------ #
# 検証・座標変換
# ------------------------------------------------------------------ #


def coverage_stats(labels: np.ndarray) -> dict:
    """完全被覆統計を計算する。"""
    arr = np.asarray(labels)
    h, w = arr.shape
    total = int(h * w)
    assigned = int(np.count_nonzero(arr > 0))
    k = int(arr.max()) if arr.size else 0
    areas = region_areas(arr, k)[1:] if k >= 1 else np.zeros(0)
    nonzero = areas[areas > 0]
    return {
        "total_pixels": total,
        "assigned_pixels": assigned,
        "unassigned_pixels": total - assigned,
        "coverage_ratio": (assigned / total) if total else 0.0,
        "overlap_pixels": 0,  # ラベルは単写像なので重複は構造的に 0
        "leaf_region_count": int(nonzero.size),
        "minimum_region_area": int(nonzero.min()) if nonzero.size else 0,
        "maximum_region_area": int(nonzero.max()) if nonzero.size else 0,
        "median_region_area": int(np.median(nonzero)) if nonzero.size else 0,
    }


def validate_base_labels(labels: np.ndarray, height: int, width: int,
                         *, check_connectivity: bool = True) -> None:
    """
    基礎ラベルが完全被覆要件を満たすか検証する。不正なら BaseLabelError。

    - shape == (height, width)
    - dtype が int32/uint32
    - すべて > 0 (未所属 0・負値なし)
    - 連続ラベル 1..K (歯抜けなし)
    - (任意) 各 id が 4 連結
    """
    arr = np.asarray(labels)
    if arr.shape != (int(height), int(width)):
        raise BaseLabelError(f"shape {arr.shape} != {(height, width)}")
    if arr.dtype not in (np.int32, np.uint32):
        raise BaseLabelError(f"dtype {arr.dtype} は int32/uint32 ではありません")
    if not np.all(arr > 0):
        raise BaseLabelError("region_id 0 以下 (未所属/負値) が残っています")
    uniq = np.unique(arr)
    k = int(uniq.max())
    if uniq.size != k or int(uniq[0]) != 1:
        raise BaseLabelError(f"ラベルが 1..{k} の連続値ではありません (種類 {uniq.size})")
    if check_connectivity:
        bbox = region_bboxes(arr, k)
        for rid in range(1, k + 1):
            x, y, bw, bh = (int(v) for v in bbox[rid])
            if bw == 0:
                continue
            sub = (arr[y:y + bh, x:x + bw] == rid).astype(np.uint8)
            ncomp, _ = cv2.connectedComponents(sub, connectivity=4)
            if ncomp > 2:
                raise BaseLabelError(f"region {rid} が非連結です ({ncomp - 1} 成分)")


def upscale_labels_nearest(labels: np.ndarray, orig_w: int, orig_h: int) -> np.ndarray:
    """作業解像度ラベルを元解像度へ最近傍で戻す。"""
    arr = np.asarray(labels)
    if arr.shape == (int(orig_h), int(orig_w)):
        return arr.astype(np.int32)
    resized = cv2.resize(
        arr.astype(np.int32), (int(orig_w), int(orig_h)),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(np.int32)
