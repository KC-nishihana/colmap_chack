"""
V0.9: 葉リージョンの軽量特徴量 (numpy + OpenCV のみ)。

8K 全解像度で高次元テクスチャ特徴を大量保持しない。スカラー/低次元のみ:
  面積, bbox, 重心, 平均Lab, Lab標準偏差, 平均輝度, 輝度標準偏差,
  平均グラデーション, 境界長。
隣接情報 (共有境界長・エッジ強度) は region_graph 側で扱う。

返却はすべて index = region_id の配列 (index 0 は未使用)。
"""

from __future__ import annotations

import numpy as np

from partition_backend import base_partition as bp

__all__ = ["LeafFeatures", "compute_leaf_features", "texture_vector"]


class LeafFeatures:
    """葉リージョン特徴量のコンテナ (index = region_id, 0 未使用)。"""

    __slots__ = (
        "k", "area", "bbox", "centroid", "mean_lab", "std_lab",
        "mean_lum", "std_lum", "mean_grad", "boundary_length",
    )

    def __init__(self, k, area, bbox, centroid, mean_lab, std_lab,
                 mean_lum, std_lum, mean_grad, boundary_length):
        self.k = k
        self.area = area
        self.bbox = bbox
        self.centroid = centroid
        self.mean_lab = mean_lab
        self.std_lab = std_lab
        self.mean_lum = mean_lum
        self.std_lum = std_lum
        self.mean_grad = mean_grad
        self.boundary_length = boundary_length


def _mean_std(labels_flat, values_flat, k):
    """region ごとの平均と標準偏差 (1D values)。"""
    area = np.bincount(labels_flat, minlength=k + 1).astype(np.float64)
    safe = np.where(area == 0, 1.0, area)
    s = np.bincount(labels_flat, weights=values_flat, minlength=k + 1)
    s2 = np.bincount(labels_flat, weights=values_flat * values_flat, minlength=k + 1)
    mean = s / safe
    var = np.maximum(s2 / safe - mean * mean, 0.0)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def _boundary_length(labels, k):
    """各 region の境界画素数 (4 近傍で異なる label または画像端に接する)。"""
    arr = labels
    h, w = arr.shape
    boundary = np.zeros((h, w), dtype=bool)
    boundary[:, :-1] |= arr[:, :-1] != arr[:, 1:]
    boundary[:, 1:] |= arr[:, :-1] != arr[:, 1:]
    boundary[:-1, :] |= arr[:-1, :] != arr[1:, :]
    boundary[1:, :] |= arr[:-1, :] != arr[1:, :]
    boundary[0, :] = True
    boundary[-1, :] = True
    boundary[:, 0] = True
    boundary[:, -1] = True
    bflat = arr[boundary].reshape(-1).astype(np.int64)
    return np.bincount(bflat, minlength=k + 1).astype(np.int64)


def compute_leaf_features(labels: np.ndarray, lab: np.ndarray,
                          grad: np.ndarray, k: int | None = None) -> LeafFeatures:
    """葉ラベル (1..K) と Lab / 勾配から軽量特徴量を計算する。"""
    arr = np.asarray(labels)
    if k is None:
        k = int(arr.max())
    flat = arr.reshape(-1).astype(np.int64)
    lab_flat = np.asarray(lab).reshape(-1, 3).astype(np.float64)
    grad_flat = np.asarray(grad).reshape(-1).astype(np.float64)

    area = bp.region_areas(arr, k)
    bbox = bp.region_bboxes(arr, k)

    # 平均/標準偏差 Lab (チャンネルごと)
    mean_lab = np.zeros((k + 1, 3), dtype=np.float32)
    std_lab = np.zeros((k + 1, 3), dtype=np.float32)
    for c in range(3):
        m, s = _mean_std(flat, lab_flat[:, c], k)
        mean_lab[:, c] = m
        std_lab[:, c] = s
    mean_lum = mean_lab[:, 0].copy()
    std_lum = std_lab[:, 0].copy()
    mean_grad, _ = _mean_std(flat, grad_flat, k)

    # 重心
    ys, xs = np.indices(arr.shape)
    safe = np.where(area == 0, 1, area).astype(np.float64)
    cx = np.bincount(flat, weights=xs.reshape(-1).astype(np.float64), minlength=k + 1) / safe
    cy = np.bincount(flat, weights=ys.reshape(-1).astype(np.float64), minlength=k + 1) / safe
    centroid = np.stack([cx, cy], axis=1).astype(np.float32)

    boundary_length = _boundary_length(arr, k)

    return LeafFeatures(
        k=k, area=area, bbox=bbox, centroid=centroid,
        mean_lab=mean_lab, std_lab=std_lab,
        mean_lum=mean_lum, std_lum=std_lum, mean_grad=mean_grad,
        boundary_length=boundary_length,
    )


def texture_vector(feat: LeafFeatures) -> np.ndarray:
    """
    node_texture 用の 4 次元軽量テクスチャ特徴 (index = region_id)。
    [輝度標準偏差, 平均グラデーション, a標準偏差, b標準偏差]。
    """
    k = feat.k
    out = np.zeros((k + 1, 4), dtype=np.float32)
    out[:, 0] = feat.std_lum
    out[:, 1] = feat.mean_grad
    out[:, 2] = feat.std_lab[:, 1]
    out[:, 3] = feat.std_lab[:, 2]
    return out
