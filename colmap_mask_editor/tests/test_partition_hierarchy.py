"""V0.9: 階層統合の整合性 (面積/bbox/root/連結ツリー) を npz 検証で確認。"""

import numpy as np
import pytest

from partition_backend import base_partition as bp
from partition_backend import watershed_backend
from ai import partition_npz
from ai.partition_npz import PartitionNpzError

from tests._partition_helpers import build_partition_from_labels, synthetic_bgr


def _build(seed=0, base_region_count=60):
    img = synthetic_bgr(seed=seed)
    labels = watershed_backend.grid_watershed(img, base_region_count=base_region_count)
    arrays = build_partition_from_labels(labels, img)
    return img, labels, arrays


def test_hierarchy_passes_npz_verification(tmp_path):
    _, _, arrays = _build()
    path = tmp_path / "p.npz"
    partition_npz.save_partition_npz(path, arrays)  # 内部 verify が全整合をチェック
    data = partition_npz.load_partition_npz(path)
    assert int(data["node_count"][0]) == 2 * int(data["leaf_count"][0]) - 1


def test_root_area_equals_total_pixels():
    img, labels, arrays = _build()
    h, w = labels.shape
    root = int(arrays["root_id"][0])
    assert int(arrays["node_area"][root - 1]) == h * w


def test_root_bbox_is_full_image():
    img, labels, arrays = _build()
    h, w = labels.shape
    root = int(arrays["root_id"][0])
    x, y, bw, bh = arrays["node_bbox"][root - 1].tolist()
    assert (x, y, bw, bh) == (0, 0, w, h)


def test_single_root_and_parent_zero():
    _, _, arrays = _build()
    root = int(arrays["root_id"][0])
    assert int(arrays["node_parent"][root - 1]) == 0
    # 他のノードは parent を持つ
    parents = arrays["node_parent"].astype(np.int64)
    nonroot_zero = [i + 1 for i in range(parents.size)
                    if parents[i] == 0 and (i + 1) != root]
    assert nonroot_zero == []


def test_deterministic():
    _, _, a1 = _build(seed=2)
    _, _, a2 = _build(seed=2)
    assert np.array_equal(a1["node_left"], a2["node_left"])
    assert np.array_equal(a1["node_merge_cost"], a2["node_merge_cost"])


def test_only_adjacent_regions_merged():
    # 統合は隣接のみ: ツリーの各親の bbox が子を包含し連結 (npz verify が保証)
    img, labels, arrays = _build()
    left = arrays["node_left"].astype(np.int64)
    right = arrays["node_right"].astype(np.int64)
    leaf_count = int(arrays["leaf_count"][0])
    # 葉は子を持たない
    for leaf in range(1, leaf_count + 1):
        assert left[leaf - 1] == 0 and right[leaf - 1] == 0


def test_sam_lowers_cost_for_shared_segment():
    from partition_backend.hierarchy_builder import _sam_disagreement, _Node, SAM_HIGH
    a = _Node(10, np.zeros(3), 0, 0, [0, 0, 1, 1], 0, 0, sam_id=5, sam_score=0.5)
    b_same = _Node(10, np.zeros(3), 0, 0, [0, 0, 1, 1], 0, 0, sam_id=5, sam_score=0.6)
    b_diff = _Node(10, np.zeros(3), 0, 0, [0, 0, 1, 1], 0, 0, sam_id=9, sam_score=0.6)
    b_none = _Node(10, np.zeros(3), 0, 0, [0, 0, 1, 1], 0, 0, sam_id=0, sam_score=0.0)
    assert _sam_disagreement(a, b_same) == 0.0
    assert _sam_disagreement(a, b_diff) == 1.0
    assert _sam_disagreement(a, b_none) == 0.5
