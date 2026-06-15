"""V0.9: 完全被覆 (coverage_ratio=1.0, 未所属0, 重複0) の検証。"""

import numpy as np

from partition_backend import base_partition as bp
from partition_backend import watershed_backend


def _img(h, w, seed):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, size=(8, 8, 3)).astype(np.uint8)
    img = np.kron(base, np.ones((h // 8 + 1, w // 8 + 1, 1), dtype=np.uint8))[:h, :w]
    noise = rng.integers(-10, 11, size=(h, w, 3))
    return np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)


def test_coverage_full_various_sizes():
    for (h, w, seed) in [(64, 64, 1), (100, 130, 2), (200, 150, 3)]:
        img = _img(h, w, seed)
        labels = watershed_backend.grid_watershed(img, base_region_count=h * w // 400)
        stats = bp.coverage_stats(labels)
        assert stats["coverage_ratio"] == 1.0, (h, w)
        assert stats["unassigned_pixels"] == 0
        assert stats["overlap_pixels"] == 0
        assert stats["leaf_region_count"] == int(labels.max())


def test_every_pixel_assigned_exactly_once():
    img = _img(120, 90, 5)
    labels = watershed_backend.grid_watershed(img, base_region_count=40)
    # 単写像なので各画素は厳密に 1 つの region。area 合計 == 画素数。
    areas = bp.region_areas(labels)
    assert int(areas.sum()) == 120 * 90
    assert int(areas[1:].sum()) == 120 * 90  # 0 は存在しない


def test_downscale_then_upscale_keeps_coverage():
    img = _img(400, 300, 7)
    ww, wh = bp.compute_working_size(300, 400, 128)
    small = bp.region_areas  # noqa (ensure import used)
    import cv2
    work = cv2.resize(img, (ww, wh), interpolation=cv2.INTER_AREA)
    labels = watershed_backend.grid_watershed(work, base_region_count=30)
    up = bp.upscale_labels_nearest(labels, 300, 400)
    bp.validate_base_labels(up, 400, 300, check_connectivity=False)
    stats = bp.coverage_stats(up)
    assert stats["coverage_ratio"] == 1.0
    assert stats["unassigned_pixels"] == 0
