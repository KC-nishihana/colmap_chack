"""
V0.9: Grid Watershed 基礎分割 (OpenCV 標準機能のみ・必須代替バックエンド)。

cv2.ximgproc が無くても動作する。処理:
  画像を作業解像度へ縮小 -> Lab 変換 -> 勾配画像 -> 規則的グリッドシード ->
  cv2.watershed -> 境界画素(-1)を隣接領域へ割当 -> 小領域統合 -> region_id 確定。

watershed が出力する -1 境界は残さない。境界画素は隣接リージョンのうち
色差が小さい -> region_id が小さい の優先順で割り当てる (決定的)。
"""

from __future__ import annotations

import cv2
import numpy as np

from partition_backend import base_partition as bp

__all__ = ["grid_watershed", "resolve_watershed_boundaries"]


def _seed_spacing(h: int, w: int, *, seed_spacing: int | None,
                  base_region_count: int | None) -> int:
    if seed_spacing is not None and int(seed_spacing) > 0:
        return max(2, int(seed_spacing))
    count = int(base_region_count) if base_region_count else 800
    count = max(1, count)
    spacing = int(round((h * w / count) ** 0.5))
    return max(2, spacing)


def _place_grid_seeds(h: int, w: int, spacing: int) -> np.ndarray:
    """グリッド中心へ一意のシード id を置いた markers (int32) を返す。"""
    markers = np.zeros((h, w), dtype=np.int32)
    ys = np.arange(spacing // 2, h, spacing)
    xs = np.arange(spacing // 2, w, spacing)
    if ys.size == 0:
        ys = np.array([h // 2])
    if xs.size == 0:
        xs = np.array([w // 2])
    seed = 1
    for y in ys:
        for x in xs:
            markers[int(y), int(x)] = seed
            seed += 1
    return markers


def resolve_watershed_boundaries(markers: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """
    watershed の -1 境界画素を隣接ラベルへ割り当てる (色差小 -> id 小)。

    完全にベクトル化したパスを繰り返し、全 -1 が解消するまで処理する。
    """
    labels = np.asarray(markers).astype(np.int32).copy()
    lab = np.asarray(lab).astype(np.float32)
    h, w = labels.shape
    INF = np.float32(np.inf)
    BIG = np.int32(np.iinfo(np.int32).max)

    # 0 (未確定) も -1 と同様に扱う (シードが疎な隅など)
    shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for _ in range(h + w):  # 上限 (収束保証)
        unknown = labels <= 0
        if not unknown.any():
            break
        best_dist = np.full((h, w), INF, dtype=np.float32)
        best_id = np.full((h, w), BIG, dtype=np.int32)
        for dy, dx in shifts:
            n_lab = np.zeros((h, w), dtype=np.int32)
            n_col = np.zeros((h, w, 3), dtype=np.float32)
            # シフト元のスライス範囲
            ys0 = max(0, -dy); ys1 = h - max(0, dy)
            xs0 = max(0, -dx); xs1 = w - max(0, dx)
            yd0 = max(0, dy); yd1 = h - max(0, -dy)
            xd0 = max(0, dx); xd1 = w - max(0, -dx)
            n_lab[yd0:yd1, xd0:xd1] = labels[ys0:ys1, xs0:xs1]
            n_col[yd0:yd1, xd0:xd1] = lab[ys0:ys1, xs0:xs1]
            valid = unknown & (n_lab > 0)
            if not valid.any():
                continue
            dist = np.linalg.norm(lab - n_col, axis=2).astype(np.float32)
            better = valid & (
                (dist < best_dist)
                | ((dist == best_dist) & (n_lab < best_id))
            )
            best_dist[better] = dist[better]
            best_id[better] = n_lab[better]
        assign = best_id < BIG
        if not assign.any():
            break  # どの -1 も labeled 隣接が無い (理論上起きない)
        labels[assign] = best_id[assign]
    if np.any(labels <= 0):
        # 念のため: 残った未確定を最近傍ラベルで埋める (決定的)
        labels[labels <= 0] = 1
    return labels


def grid_watershed(
    image_bgr: np.ndarray,
    *,
    seed_spacing: int | None = None,
    base_region_count: int | None = None,
    min_area: int = 0,
    enforce_connectivity: bool = True,
) -> np.ndarray:
    """
    作業解像度の BGR 画像から完全被覆ラベル (1..K, int32) を生成する。

    呼び出し側で作業解像度へ縮小済みの画像を渡すこと。返り値は同解像度のラベル。
    """
    img = np.asarray(image_bgr)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = np.ascontiguousarray(img[..., :3].astype(np.uint8))
    h, w = img.shape[:2]
    lab = bp.to_lab(img)

    spacing = _seed_spacing(h, w, seed_spacing=seed_spacing,
                            base_region_count=base_region_count)
    markers = _place_grid_seeds(h, w, spacing)
    cv2.watershed(img, markers)  # markers を in-place 更新 (境界 = -1)

    labels = resolve_watershed_boundaries(markers, lab)
    labels = bp.relabel_sequential(labels)
    if enforce_connectivity:
        labels = bp.enforce_connectivity(labels)
    if min_area and min_area > 1:
        labels = bp.merge_small_regions(labels, lab, min_area)
    return bp.relabel_sequential(labels).astype(np.int32)
