"""V0.9: Region Adjacency Graph と葉特徴量 + SAM シグネチャ。"""

import numpy as np

from partition_backend import base_partition as bp
from partition_backend import region_features, region_graph, sam_guidance
from ai import amg_npz, amg_rle


def _three_stripes():
    labels = np.zeros((6, 9), dtype=np.int32)
    labels[:, 0:3] = 1
    labels[:, 3:6] = 2
    labels[:, 6:9] = 3
    return labels


def test_rag_4_adjacency_only():
    labels = _three_stripes()
    grad = np.ones((6, 9), dtype=np.float32)
    g = region_graph.build_region_graph(labels, grad, k=3)
    edges = {(a, b) for a, b, _, _ in g.edges()}
    assert edges == {(1, 2), (2, 3)}  # 1 と 3 は隣接しない
    assert (1, 3) not in edges


def test_rag_no_duplicate_pairs():
    labels = _three_stripes()
    grad = np.zeros((6, 9), dtype=np.float32)
    g = region_graph.build_region_graph(labels, grad, k=3)
    pairs = list(zip(g.edge_a.tolist(), g.edge_b.tolist()))
    assert len(pairs) == len(set(pairs))


def test_rag_shared_boundary_length():
    labels = _three_stripes()
    grad = np.zeros((6, 9), dtype=np.float32)
    g = region_graph.build_region_graph(labels, grad, k=3)
    adj = {(a, b): L for a, b, L, _ in g.edges()}
    # 1|2 境界は縦 6 画素分接する
    assert adj[(1, 2)] == 6
    assert adj[(2, 3)] == 6


def test_rag_diagonal_not_adjacent():
    labels = np.array([
        [1, 2],
        [2, 1],
    ], dtype=np.int32)
    # ここでは 1 と 2 は上下左右でも接触する
    g = region_graph.build_region_graph(labels, np.zeros((2, 2), np.float32), k=2)
    # 純粋対角のみ: 3x3 で 1 を対角、残りを背景にしない構成
    labels2 = np.array([
        [1, 3],
        [3, 1],
    ], dtype=np.int32)
    # 1 は (0,0),(1,1); 3 は (0,1),(1,0)。4 近傍で 1-3 は接触する。
    g2 = region_graph.build_region_graph(labels2, np.zeros((2, 2), np.float32), k=3)
    assert g2.num_edges >= 1


def test_leaf_features():
    labels = _three_stripes()
    lab = np.zeros((6, 9, 3), dtype=np.float32)
    lab[:, 0:3, 0] = 10.0
    lab[:, 3:6, 0] = 100.0
    lab[:, 6:9, 0] = 200.0
    grad = np.zeros((6, 9), dtype=np.float32)
    feat = region_features.compute_leaf_features(labels, lab, grad, k=3)
    assert feat.area[1] == 18 and feat.area[2] == 18 and feat.area[3] == 18
    assert abs(feat.mean_lab[1, 0] - 10.0) < 1e-3
    assert abs(feat.mean_lab[3, 0] - 200.0) < 1e-3
    # 重心 x: stripe1 中央 ~1, stripe3 ~7
    assert abs(feat.centroid[1, 0] - 1.0) < 1e-3
    assert abs(feat.centroid[3, 0] - 7.0) < 1e-3
    tex = region_features.texture_vector(feat)
    assert tex.shape == (4, 4)


# ------------------------------------------------------------------ #
# SAM シグネチャ
# ------------------------------------------------------------------ #


def _fake_segments(h, w, masks_with_meta):
    """masks_with_meta: [(mask_bool, iou, stab)] -> in-memory segments dict。"""
    anns = []
    for mask, iou, stab in masks_with_meta:
        counts = amg_rle.encode_mask(mask.astype(np.uint8))
        ys, xs = np.where(mask)
        x0, y0 = int(xs.min()), int(ys.min())
        bw = int(xs.max() - xs.min() + 1)
        bh = int(ys.max() - ys.min() + 1)
        anns.append({
            "segmentation": {"size": [h, w], "counts": counts},
            "bbox": [x0, y0, bw, bh],
            "area": int(mask.sum()),
            "predicted_iou": iou,
            "stability_score": stab,
            "point_coords": [[float(x0), float(y0)]],
        })
    return amg_npz.build_segment_arrays(anns, h, w)


def test_sam_signature_basic():
    labels = _three_stripes()  # 6x9, leaves 1..3
    h, w = labels.shape
    # SAM 候補: stripe1 を覆うマスク (高信頼)
    m1 = np.zeros((h, w), dtype=bool)
    m1[:, 0:3] = True
    data = _fake_segments(h, w, [(m1, 0.9, 0.95)])
    sig = sam_guidance.compute_sam_signatures(labels, data, sample_count=16, top_k=4)
    ids1, cov1, sc1 = sig.for_leaf(1)
    assert ids1.size == 1
    assert cov1[0] > 0.99  # stripe1 完全被覆
    # stripe3 は交差しないので空
    ids3, _, _ = sig.for_leaf(3)
    assert ids3.size == 0


def test_sam_signature_none_continues():
    labels = _three_stripes()
    sig = sam_guidance.compute_sam_signatures(labels, None)
    assert sig.segment_ids.size == 0
    assert sig.offsets.shape == (4,)


def test_sam_signature_topk_limit():
    labels = _three_stripes()
    h, w = labels.shape
    masks = []
    for i in range(6):
        m = np.zeros((h, w), dtype=bool)
        m[:, 0:3] = True
        masks.append((m, 0.8 + i * 0.01, 0.9))
    data = _fake_segments(h, w, masks)
    sig = sam_guidance.compute_sam_signatures(labels, data, sample_count=16, top_k=4)
    ids1, _, _ = sig.for_leaf(1)
    assert ids1.size == 4  # top_k で制限
