"""
V0.7: 伝播結果マスクの品質指標と警告判定 (numpy のみ・torch非依存)。

品質指標は自動採否ではなく警告表示に使う。カメラ移動が大きいと正しい対象でも
IoU や重心が大きく変化しうるため、V0.7 では警告だけで自動破棄しない。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# 警告コード
class WarnCode:
    EMPTY_MASK = "EMPTY_MASK"
    TOO_LARGE = "TOO_LARGE"
    AREA_DROP = "AREA_DROP"
    AREA_GROWTH = "AREA_GROWTH"
    MANY_COMPONENTS = "MANY_COMPONENTS"
    TOUCHES_ALL_EDGES = "TOUCHES_ALL_EDGES"
    LOW_IOU = "LOW_IOU"


@dataclass(frozen=True)
class QualityThresholds:
    too_large_ratio: float = 0.80      # 前景率がこれ以上で TOO_LARGE
    area_drop_ratio: float = 0.25      # 前フレーム比これ未満で AREA_DROP
    area_growth_ratio: float = 4.00    # 前フレーム比これ超過で AREA_GROWTH
    component_count: int = 10          # 連結成分がこれ超で MANY_COMPONENTS
    low_iou: float = 0.05              # 前フレームとの IoU これ未満で LOW_IOU


@dataclass
class FrameMetrics:
    foreground_pixels: int = 0
    foreground_ratio: float = 0.0
    bbox: Optional[tuple[int, int, int, int]] = None  # (x0,y0,x1,y1) 包含
    centroid: Optional[tuple[float, float]] = None
    component_count: int = 0
    touches_top: bool = False
    touches_bottom: bool = False
    touches_left: bool = False
    touches_right: bool = False
    area_ratio_to_prev: Optional[float] = None
    iou_to_prev: Optional[float] = None
    warning_codes: list[str] = field(default_factory=list)


def _connected_components(binary: np.ndarray) -> int:
    """8近傍の連結成分数。cv2 があれば使い、無ければ簡易ラベリング。"""
    try:
        import cv2
        num, _ = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
        return max(0, num - 1)  # 背景ラベル0を除く
    except Exception:
        return _cc_fallback(binary)


def _cc_fallback(binary: np.ndarray) -> int:
    # 依存なしの BFS ラベリング (テスト用フォールバック)。
    h, w = binary.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    from collections import deque
    for y in range(h):
        for x in range(w):
            if binary[y, x] and not seen[y, x]:
                count += 1
                dq = deque([(y, x)])
                seen[y, x] = True
                while dq:
                    cy, cx = dq.popleft()
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and not seen[ny, nx]:
                                seen[ny, nx] = True
                                dq.append((ny, nx))
    return count


def compute_metrics(
    mask: np.ndarray,
    prev_mask: Optional[np.ndarray] = None,
    thresholds: QualityThresholds = QualityThresholds(),
) -> FrameMetrics:
    """
    mask: uint8 (H,W) 0/255 (または bool)。前フレーム prev_mask があれば面積比/IoUも算出。
    """
    if mask.ndim != 2:
        raise ValueError(f"mask は2次元である必要があります: shape={mask.shape}")
    fg = mask > 0
    h, w = fg.shape
    total = h * w
    n_fg = int(fg.sum())

    m = FrameMetrics(
        foreground_pixels=n_fg,
        foreground_ratio=(n_fg / total) if total else 0.0,
    )

    if n_fg == 0:
        m.warning_codes.append(WarnCode.EMPTY_MASK)
        if prev_mask is not None:
            m.area_ratio_to_prev = 0.0 if int((prev_mask > 0).sum()) > 0 else None
            m.iou_to_prev = 0.0 if int((prev_mask > 0).sum()) > 0 else None
            if m.iou_to_prev is not None and m.iou_to_prev < thresholds.low_iou:
                m.warning_codes.append(WarnCode.LOW_IOU)
        return m

    ys, xs = np.where(fg)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    m.bbox = (x0, y0, x1, y1)
    m.centroid = (float(xs.mean()), float(ys.mean()))
    m.component_count = _connected_components(fg)

    m.touches_top = bool(fg[0, :].any())
    m.touches_bottom = bool(fg[h - 1, :].any())
    m.touches_left = bool(fg[:, 0].any())
    m.touches_right = bool(fg[:, w - 1].any())

    if prev_mask is not None:
        prev_fg = prev_mask > 0
        n_prev = int(prev_fg.sum())
        if n_prev > 0:
            m.area_ratio_to_prev = n_fg / n_prev
            inter = int(np.logical_and(fg, prev_fg).sum())
            union = int(np.logical_or(fg, prev_fg).sum())
            m.iou_to_prev = (inter / union) if union else 0.0

    _apply_warnings(m, thresholds)
    return m


def _apply_warnings(m: FrameMetrics, t: QualityThresholds) -> None:
    if m.foreground_ratio >= t.too_large_ratio:
        m.warning_codes.append(WarnCode.TOO_LARGE)
    if m.component_count > t.component_count:
        m.warning_codes.append(WarnCode.MANY_COMPONENTS)
    if m.touches_top and m.touches_bottom and m.touches_left and m.touches_right:
        m.warning_codes.append(WarnCode.TOUCHES_ALL_EDGES)
    if m.area_ratio_to_prev is not None:
        if m.area_ratio_to_prev < t.area_drop_ratio:
            m.warning_codes.append(WarnCode.AREA_DROP)
        elif m.area_ratio_to_prev > t.area_growth_ratio:
            m.warning_codes.append(WarnCode.AREA_GROWTH)
    if m.iou_to_prev is not None and m.iou_to_prev < t.low_iou:
        m.warning_codes.append(WarnCode.LOW_IOU)
