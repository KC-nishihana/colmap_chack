"""V0.9 partition テスト用の小さな決定的フィクスチャ生成ヘルパー (numpy のみ)。"""

from __future__ import annotations

import numpy as np

from ai import partition_npz, partition_rle


def make_partition_arrays_from_labels(
    labels: np.ndarray,
    merges: list[tuple[int, int]],
    *,
    sam: dict[int, list[tuple[int, float, float]]] | None = None,
) -> dict[str, np.ndarray]:
    """
    葉ラベルマップ (値 1..K) と統合順 (葉/親 node_id のペア列) から
    partition.npz 配列を組み立てる。テスト専用の決定的ビルダー。

    merges: [(node_a, node_b), ...] の順で親ノード K+1, K+2, ... を作る。
            最後の統合が root。各 node_id の area/bbox/centroid は labels から算出。
    sam: {leaf_id: [(segment_id, coverage, score), ...]} (任意)。
    """
    arr = np.asarray(labels)
    h, w = arr.shape
    leaf_ids = np.unique(arr)
    leaf_count = int(leaf_ids.max())
    assert leaf_ids[0] == 1 and leaf_count == leaf_ids.size, "labels must be 1..K dense"

    node_count = leaf_count + len(merges)
    # 葉の幾何特徴を算出
    area = np.zeros(node_count, dtype=np.int64)
    bbox = np.zeros((node_count, 4), dtype=np.int64)  # x,y,w,h
    cxsum = np.zeros(node_count, dtype=np.float64)
    cysum = np.zeros(node_count, dtype=np.float64)
    for lid in range(1, leaf_count + 1):
        ys, xs = np.where(arr == lid)
        area[lid - 1] = xs.size
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        bbox[lid - 1] = [x0, y0, x1 - x0 + 1, y1 - y0 + 1]
        cxsum[lid - 1] = float(xs.sum())
        cysum[lid - 1] = float(ys.sum())

    left = np.zeros(node_count, dtype=np.int64)
    right = np.zeros(node_count, dtype=np.int64)
    parent = np.zeros(node_count, dtype=np.int64)
    merge_cost = np.zeros(node_count, dtype=np.float64)
    level = np.zeros(node_count, dtype=np.int64)

    next_id = leaf_count + 1
    for i, (a, b) in enumerate(merges):
        nid = next_id + i
        left[nid - 1] = a
        right[nid - 1] = b
        parent[a - 1] = nid
        parent[b - 1] = nid
        area[nid - 1] = area[a - 1] + area[b - 1]
        ax, ay, aw, ah = bbox[a - 1]
        bx, by, bw, bh = bbox[b - 1]
        x0 = min(ax, bx)
        y0 = min(ay, by)
        x1 = max(ax + aw, bx + bw)
        y1 = max(ay + ah, by + bh)
        bbox[nid - 1] = [x0, y0, x1 - x0, y1 - y0]
        cxsum[nid - 1] = cxsum[a - 1] + cxsum[b - 1]
        cysum[nid - 1] = cysum[a - 1] + cysum[b - 1]
        merge_cost[nid - 1] = 0.1 * (i + 1)
        level[nid - 1] = max(level[a - 1], level[b - 1]) + 1
    root_id = node_count

    centroid = np.zeros((node_count, 2), dtype=np.float32)
    for nid in range(1, node_count + 1):
        a = max(int(area[nid - 1]), 1)
        centroid[nid - 1] = [cxsum[nid - 1] / a, cysum[nid - 1] / a]

    run_ids, run_len = partition_rle.encode_label_map(arr)

    # SAM シグネチャ
    sam = sam or {}
    sam_seg: list[int] = []
    sam_cov: list[float] = []
    sam_sco: list[float] = []
    offsets = np.zeros(leaf_count + 1, dtype=np.uint64)
    for lid in range(1, leaf_count + 1):
        for seg_id, cov, sco in sam.get(lid, []):
            sam_seg.append(seg_id)
            sam_cov.append(cov)
            sam_sco.append(sco)
        offsets[lid] = len(sam_seg)

    return partition_npz.build_partition_arrays(
        height=h, width=w,
        run_region_ids=run_ids, run_lengths=run_len,
        leaf_count=leaf_count,
        node_left=left, node_right=right, node_parent=parent,
        node_area=area, node_bbox=bbox, node_centroid=centroid,
        node_mean_lab=np.zeros((node_count, 3), dtype=np.float32),
        node_texture=np.zeros((node_count, 4), dtype=np.float32),
        node_merge_cost=merge_cost, node_level=level,
        root_id=root_id,
        sam_sig_offsets=offsets,
        sam_segment_ids=np.asarray(sam_seg, dtype=np.uint32),
        sam_coverages=np.asarray(sam_cov, dtype=np.float32),
        sam_scores=np.asarray(sam_sco, dtype=np.float32),
    )


def build_partition_from_labels(
    labels: np.ndarray,
    image_bgr: np.ndarray,
    segments_data: dict | None = None,
    *,
    weights=None,
    sample_count: int = 32,
    top_k: int = 4,
) -> dict[str, np.ndarray]:
    """ラベル + 画像から features/graph/sam/hierarchy を実行し partition 配列を返す。"""
    from partition_backend import base_partition as bp
    from partition_backend import region_features, region_graph, sam_guidance
    from partition_backend import hierarchy_builder as hb
    from ai import partition_npz, partition_rle

    arr = np.asarray(labels).astype(np.int32)
    h, w = arr.shape
    lab = bp.to_lab(image_bgr)
    grad = bp.gradient_magnitude(lab)
    k = int(arr.max())

    feat = region_features.compute_leaf_features(arr, lab, grad, k)
    graph = region_graph.build_region_graph(arr, grad, k)
    sig = (sam_guidance.compute_sam_signatures(arr, segments_data,
                                               sample_count=sample_count, top_k=top_k)
           if segments_data is not None else sam_guidance.empty_signatures(k))
    weights = weights or hb.MergeWeights()
    result = hb.build_hierarchy(feat, graph, sig, weights, h * w)

    run_ids, run_len = partition_rle.encode_label_map(arr)
    return partition_npz.build_partition_arrays(
        height=h, width=w,
        run_region_ids=run_ids, run_lengths=run_len,
        leaf_count=result.leaf_count,
        node_left=result.node_left, node_right=result.node_right,
        node_parent=result.node_parent, node_area=result.node_area,
        node_bbox=result.node_bbox, node_centroid=result.node_centroid,
        node_mean_lab=result.node_mean_lab, node_texture=result.node_texture,
        node_merge_cost=result.node_merge_cost, node_level=result.node_level,
        root_id=result.root_id,
        sam_sig_offsets=sig.offsets, sam_segment_ids=sig.segment_ids,
        sam_coverages=sig.coverages, sam_scores=sig.scores,
    )


def synthetic_bgr(h=120, w=160, seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[: h // 2, : w // 2] = (40, 40, 200)
    img[: h // 2, w // 2:] = (40, 200, 40)
    img[h // 2:, : w // 2] = (200, 40, 40)
    img[h // 2:, w // 2:] = (200, 200, 40)
    noise = rng.integers(-12, 13, size=(h, w, 3))
    return np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)


def simple_three_leaf() -> dict[str, np.ndarray]:
    """3 葉 (縦帯) の最小 partition。node 4=merge(1,2), node5=merge(4,3)=root。"""
    labels = np.zeros((4, 6), dtype=np.int32)
    labels[:, 0:2] = 1
    labels[:, 2:4] = 2
    labels[:, 4:6] = 3
    return make_partition_arrays_from_labels(labels, [(1, 2), (4, 3)])
