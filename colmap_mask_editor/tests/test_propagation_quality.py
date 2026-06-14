"""品質指標・警告判定のテスト (torch不要)。"""

import numpy as np

from ai.propagation_quality import QualityThresholds, WarnCode, compute_metrics


def _rect(w, h, x0, y0, x1, y1):
    m = np.zeros((h, w), np.uint8)
    m[y0:y1, x0:x1] = 255
    return m


def test_basic_metrics():
    m = _rect(100, 100, 10, 20, 30, 40)  # 20x20=400px
    met = compute_metrics(m)
    assert met.foreground_pixels == 400
    assert abs(met.foreground_ratio - 0.04) < 1e-6
    assert met.bbox == (10, 20, 29, 39)
    assert met.component_count == 1
    assert met.warning_codes == []


def test_empty_mask_warns():
    met = compute_metrics(np.zeros((50, 50), np.uint8))
    assert WarnCode.EMPTY_MASK in met.warning_codes
    assert met.foreground_pixels == 0


def test_too_large_warns():
    m = np.full((50, 50), 255, np.uint8)
    met = compute_metrics(m)
    assert WarnCode.TOO_LARGE in met.warning_codes
    assert WarnCode.TOUCHES_ALL_EDGES in met.warning_codes


def test_area_drop_warns():
    prev = _rect(100, 100, 0, 0, 50, 50)   # 2500
    cur = _rect(100, 100, 0, 0, 10, 10)    # 100 -> ratio 0.04 < 0.25
    met = compute_metrics(cur, prev_mask=prev)
    assert WarnCode.AREA_DROP in met.warning_codes
    assert met.area_ratio_to_prev is not None and met.area_ratio_to_prev < 0.25


def test_area_growth_warns():
    prev = _rect(100, 100, 0, 0, 10, 10)   # 100
    cur = _rect(100, 100, 0, 0, 80, 80)    # 6400 -> ratio 64 > 4
    met = compute_metrics(cur, prev_mask=prev)
    assert WarnCode.AREA_GROWTH in met.warning_codes


def test_low_iou_warns():
    prev = _rect(100, 100, 0, 0, 20, 20)
    cur = _rect(100, 100, 80, 80, 100, 100)  # 重ならない -> IoU 0
    met = compute_metrics(cur, prev_mask=prev)
    assert WarnCode.LOW_IOU in met.warning_codes
    assert met.iou_to_prev == 0.0


def test_many_components_warns():
    m = np.zeros((100, 100), np.uint8)
    # 12個の孤立点 (8近傍で分離するよう2px間隔)
    for i in range(12):
        m[2 * i, 0] = 255
    met = compute_metrics(m, thresholds=QualityThresholds(component_count=10))
    assert WarnCode.MANY_COMPONENTS in met.warning_codes
    assert met.component_count == 12


def test_high_iou_no_warning():
    prev = _rect(100, 100, 10, 10, 50, 50)
    cur = _rect(100, 100, 11, 11, 51, 51)  # ほぼ重なる
    met = compute_metrics(cur, prev_mask=prev)
    assert WarnCode.LOW_IOU not in met.warning_codes
    assert met.iou_to_prev > 0.5
