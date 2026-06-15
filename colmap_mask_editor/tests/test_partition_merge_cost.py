"""V0.9: 統合コスト (色/テクスチャ/境界/SAM/サイズ重み) と SAM ガイドの影響。"""

import numpy as np
import pytest

from partition_backend import base_partition as bp
from partition_backend import region_features, region_graph, sam_guidance
from partition_backend import hierarchy_builder as hb
from partition_backend.hierarchy_builder import MergeWeights, _sam_disagreement, _Node


def _two_region_image(color_a, color_b, h=8, w=16):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, : w // 2] = color_a
    img[:, w // 2:] = color_b
    labels = np.zeros((h, w), dtype=np.int32)
    labels[:, : w // 2] = 1
    labels[:, w // 2:] = 2
    return img, labels


def _build(img, labels, sig=None):
    lab = bp.to_lab(img)
    grad = bp.gradient_magnitude(lab)
    k = int(np.asarray(labels).max())
    feat = region_features.compute_leaf_features(labels, lab, grad, k)
    graph = region_graph.build_region_graph(labels, grad, k)
    sig = sig or sam_guidance.empty_signatures(k)
    return hb.build_hierarchy(feat, graph, sig, MergeWeights(), img.shape[0] * img.shape[1])


def test_similar_adjacent_pair_merges_first():
    # 3 帯: 帯1,2 は色が近く、帯3 は大きく異なる。最初の統合は隣接かつ
    # 最も似ている (帯1,2) であるべき (色が近いほど統合しやすい)。
    h, w = 8, 18
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, 0:6] = (100, 100, 100)
    img[:, 6:12] = (108, 104, 102)   # 帯1 に近い
    img[:, 12:18] = (10, 220, 30)    # 大きく異なる
    labels = np.zeros((h, w), dtype=np.int32)
    labels[:, 0:6] = 1
    labels[:, 6:12] = 2
    labels[:, 12:18] = 3
    r = _build(img, labels)
    # 最初に作られる親ノード = leaf_count + 1 = 4
    first_parent = 4
    children = {int(r.node_left[first_parent - 1]), int(r.node_right[first_parent - 1])}
    assert children == {1, 2}  # 似た隣接ペアが最初に統合される


def test_sam_disagreement_rules():
    a = _Node(10, np.zeros(3), 0, 0, [0, 0, 1, 1], 0, 0, sam_id=3, sam_score=0.5)
    same = _Node(10, np.zeros(3), 0, 0, [0, 0, 1, 1], 0, 0, sam_id=3, sam_score=0.4)
    diff = _Node(10, np.zeros(3), 0, 0, [0, 0, 1, 1], 0, 0, sam_id=7, sam_score=0.4)
    none = _Node(10, np.zeros(3), 0, 0, [0, 0, 1, 1], 0, 0, sam_id=0, sam_score=0.0)
    assert _sam_disagreement(a, same) == 0.0   # 同一高信頼 SAM -> 統合しやすい
    assert _sam_disagreement(a, diff) == 1.0    # 異なる高信頼 SAM -> 統合しにくい
    assert _sam_disagreement(a, none) == 0.5    # SAM 情報なし -> 中立


def test_weights_default_values():
    w = MergeWeights()
    assert (w.color, w.texture, w.boundary, w.sam, w.size) == (0.30, 0.10, 0.30, 0.25, 0.05)


def test_merge_cost_in_unit_range():
    img, labels = _two_region_image((10, 200, 10), (200, 10, 200))
    r = _build(img, labels)
    cost = float(r.node_merge_cost[r.root_id - 1])
    assert 0.0 <= cost <= 1.0  # 各項 0..1 + 重み合計 1.0 -> [0,1]
