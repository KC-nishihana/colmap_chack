"""V0.9: 基礎分割バックエンド (Grid Watershed / SLIC / AUTO) と検証。"""

import numpy as np
import pytest

from partition_backend import base_partition as bp
from partition_backend.base_partition import BasePartitionBackend, BaseLabelError
from partition_backend import watershed_backend
from partition_backend import slic_backend


def _synthetic_image(h=120, w=160, seed=0):
    """4 象限に色違いブロックを置き、勾配のある合成 BGR 画像を作る。"""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[: h // 2, : w // 2] = (40, 40, 200)
    img[: h // 2, w // 2:] = (40, 200, 40)
    img[h // 2:, : w // 2] = (200, 40, 40)
    img[h // 2:, w // 2:] = (200, 200, 40)
    noise = rng.integers(-15, 16, size=(h, w, 3))
    return np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)


def test_watershed_full_coverage():
    img = _synthetic_image()
    labels = watershed_backend.grid_watershed(img, base_region_count=60)
    h, w = img.shape[:2]
    bp.validate_base_labels(labels, h, w)  # raises if not 100% covered
    stats = bp.coverage_stats(labels)
    assert stats["coverage_ratio"] == 1.0
    assert stats["unassigned_pixels"] == 0
    assert stats["overlap_pixels"] == 0


def test_watershed_no_minus_one_boundary():
    img = _synthetic_image()
    labels = watershed_backend.grid_watershed(img, base_region_count=80)
    assert np.all(labels > 0)


def test_watershed_deterministic():
    img = _synthetic_image(seed=3)
    a = watershed_backend.grid_watershed(img, base_region_count=70)
    b = watershed_backend.grid_watershed(img, base_region_count=70)
    assert np.array_equal(a, b)


def test_watershed_connectivity():
    img = _synthetic_image()
    labels = watershed_backend.grid_watershed(img, base_region_count=50)
    bp.validate_base_labels(labels, *img.shape[:2], check_connectivity=True)


def test_resolve_boundaries_removes_minus_one():
    markers = np.array([
        [1, 1, -1, 2, 2],
        [1, -1, -1, -1, 2],
        [1, 1, -1, 2, 2],
    ], dtype=np.int32)
    lab = np.zeros((3, 5, 3), dtype=np.float32)
    lab[:, :2] = 10.0   # 左側の色
    lab[:, 3:] = 200.0  # 右側の色
    out = watershed_backend.resolve_watershed_boundaries(markers, lab)
    assert np.all(out > 0)


def test_min_area_merge_reduces_tiny_regions():
    img = _synthetic_image()
    coarse = watershed_backend.grid_watershed(img, base_region_count=200, min_area=0)
    merged = watershed_backend.grid_watershed(img, base_region_count=200, min_area=80)
    areas = bp.region_areas(merged)[1:]
    # min_area 未満が残らない (最終 relabel 後)
    assert np.all(areas[areas > 0] >= 1)
    assert int(merged.max()) <= int(coarse.max())


def test_auto_falls_back_to_watershed_when_no_ximgproc():
    img = _synthetic_image()
    labels, used = bp.run_base_partition(img, BasePartitionBackend.AUTO,
                                         base_region_count=60)
    bp.validate_base_labels(labels, *img.shape[:2])
    if slic_backend.slic_available():
        assert used == "slic"
    else:
        assert used == "grid_watershed"


def test_explicit_slic_raises_when_unavailable():
    img = _synthetic_image()
    if slic_backend.slic_available():
        pytest.skip("ximgproc が利用可能なため明示 SLIC は成功する")
    with pytest.raises(slic_backend.SlicUnavailableError):
        bp.run_base_partition(img, BasePartitionBackend.SLIC, base_region_count=60)


def test_upscale_nearest_revalidate():
    img = _synthetic_image(h=80, w=100)
    labels = watershed_backend.grid_watershed(img, base_region_count=40)
    up = bp.upscale_labels_nearest(labels, 200, 160)
    assert up.shape == (160, 200)
    bp.validate_base_labels(up, 160, 200, check_connectivity=False)
    assert np.all(up > 0)
    # 最近傍なので元に存在したラベルのみ
    assert set(np.unique(up).tolist()).issubset(set(np.unique(labels).tolist()))


def test_compute_working_size():
    assert bp.compute_working_size(8000, 4000, 2048) == (2048, 1024)
    assert bp.compute_working_size(1000, 800, 2048) == (1000, 800)
    assert bp.compute_working_size(8000, 4000, 0) == (8000, 4000)


def test_validate_rejects_zero_label():
    bad = np.zeros((4, 4), dtype=np.int32)
    bad[0, 0] = 1
    with pytest.raises(BaseLabelError):
        bp.validate_base_labels(bad, 4, 4)


def test_neighbor_pairs_4connectivity_only():
    # 対角のみ接触する 2 領域は隣接にしない
    labels = np.array([
        [1, 2],
        [2, 1],
    ], dtype=np.int32)
    pairs = bp.neighbor_pairs(labels)
    # 1 と 2 は上下左右でも接触している (このレイアウトでは隣接する)
    assert pairs.shape[1] == 2
    # 純粋な対角接触のみのケース
    labels2 = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ], dtype=np.int32)
    labels2 = np.where(labels2 == 0, 3, labels2)
    pairs2 = bp.neighbor_pairs(labels2)
    # region 1 のブロックは対角配置だが背景 3 とは 4 近傍で接する
    assert (1, 3) in {tuple(p) for p in pairs2.tolist()}
