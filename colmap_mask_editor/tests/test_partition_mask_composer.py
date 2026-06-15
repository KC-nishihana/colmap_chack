"""V0.9: 最終マスク生成 (KEEP=255 / REMOVE=0, C-order, クリック一致)。"""

import numpy as np
import pytest

from ai import partition_mask_composer as pmc
from ai.partition_hit_test import PartitionHitTester

from tests._partition_helpers import simple_three_leaf

ARR = simple_three_leaf()
PARENT = ARR["node_parent"]


def test_keep_is_255_remove_is_0():
    lut = pmc.leaf_decision_values(PARENT, 3, {"1": "keep", "2": "remove", "3": "keep"})
    mask = pmc.compose_mask(ARR["run_region_ids"], ARR["run_lengths"], 4, 6, lut)
    assert mask.shape == (4, 6)
    assert mask.dtype == np.uint8
    # leaf1 (cols0-1)=255, leaf2(cols2-3)=0, leaf3(cols4-5)=255
    assert np.all(mask[:, 0:2] == 255)
    assert np.all(mask[:, 2:4] == 0)
    assert np.all(mask[:, 4:6] == 255)


def test_unreviewed_raises_without_action():
    with pytest.raises(ValueError):
        pmc.leaf_decision_values(PARENT, 3, {"1": "keep"})  # 2,3 未確認


def test_unreviewed_as_remove():
    lut = pmc.leaf_decision_values(PARENT, 3, {"1": "keep"}, unreviewed_as="remove")
    mask = pmc.compose_mask(ARR["run_region_ids"], ARR["run_lengths"], 4, 6, lut)
    assert np.all(mask[:, 0:2] == 255)
    assert np.all(mask[:, 2:6] == 0)


def test_all_pixels_decided():
    lut = pmc.leaf_decision_values(PARENT, 3, {"5": "keep"})
    mask = pmc.compose_mask(ARR["run_region_ids"], ARR["run_lengths"], 4, 6, lut)
    assert np.all((mask == 0) | (mask == 255))
    assert np.all(mask == 255)


def test_click_position_matches_output_pixel():
    # 各画素のクリック葉判定と最終マスク値が一致する
    lut = pmc.leaf_decision_values(PARENT, 3, {"1": "keep", "2": "remove", "3": "keep"})
    mask = pmc.compose_mask(ARR["run_region_ids"], ARR["run_lengths"], 4, 6, lut)
    ht = PartitionHitTester(ARR)
    leaf_to_val = {1: 255, 2: 0, 3: 255}
    for y in range(4):
        for x in range(6):
            leaf = ht.leaf_at(x, y)
            assert mask[y, x] == leaf_to_val[leaf], (x, y)


def test_save_mask_png_atomic(tmp_path):
    lut = pmc.leaf_decision_values(PARENT, 3, {"5": "keep"})
    mask = pmc.compose_mask(ARR["run_region_ids"], ARR["run_lengths"], 4, 6, lut)
    path = tmp_path / "サブ フォルダ" / "mask.png"  # 日本語 + 全角スペース
    pmc.save_mask_png(path, mask)
    import cv2
    loaded = cv2.imdecode(
        np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    assert loaded.shape == (4, 6)
    assert np.array_equal(loaded, mask)
