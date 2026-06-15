"""
V0.9: SLICO 基礎分割 (OpenCV contrib の cv2.ximgproc を使用)。

ximgproc が無い環境では SlicUnavailableError を送出する。AUTO バックエンドの
場合だけ呼び出し側が Grid Watershed へフォールバックする。ユーザーが SLIC を
明示指定したのに ximgproc が無い場合は、このエラーをそのまま表示する。
"""

from __future__ import annotations

import cv2
import numpy as np

from partition_backend import base_partition as bp

__all__ = ["slic_available", "slic_superpixels", "SlicUnavailableError"]


class SlicUnavailableError(RuntimeError):
    """cv2.ximgproc.createSuperpixelSLIC が利用できないときに送出する。"""


def slic_available() -> bool:
    """cv2.ximgproc の SLIC が利用可能か。"""
    return hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "createSuperpixelSLIC")


def slic_superpixels(
    image_bgr: np.ndarray,
    *,
    region_size: int | None = None,
    ruler: float = 10.0,
    base_region_count: int | None = None,
    min_area: int = 0,
    iterations: int = 10,
    enforce_connectivity: bool = True,
) -> np.ndarray:
    """
    作業解像度 BGR 画像から SLICO 基礎ラベル (1..K, int32) を生成する。

    enforceLabelConnectivity 済みのため各 superpixel は連結。
    """
    if not slic_available():
        raise SlicUnavailableError(
            "cv2.ximgproc.createSuperpixelSLIC が見つかりません "
            "(opencv-contrib-python が必要です)。"
        )
    img = np.asarray(image_bgr)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = np.ascontiguousarray(img[..., :3].astype(np.uint8))
    h, w = img.shape[:2]
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    if region_size is None or int(region_size) <= 0:
        count = max(1, int(base_region_count) if base_region_count else 800)
        region_size = max(4, int(round((h * w / count) ** 0.5)))

    slic = cv2.ximgproc.createSuperpixelSLIC(
        lab_img,
        algorithm=cv2.ximgproc.SLICO,
        region_size=int(region_size),
        ruler=float(ruler),
    )
    slic.iterate(int(iterations))
    min_elem = max(1, int(region_size * region_size) // 4)
    slic.enforceLabelConnectivity(min_elem)
    labels = slic.getLabels().astype(np.int32)  # 0..N-1

    labels = bp.relabel_sequential(labels)
    if enforce_connectivity:
        labels = bp.enforce_connectivity(labels)
    if min_area and min_area > 1:
        lab_f = lab_img.astype(np.float32)
        labels = bp.merge_small_regions(labels, lab_f, min_area)
    return bp.relabel_sequential(labels).astype(np.int32)
