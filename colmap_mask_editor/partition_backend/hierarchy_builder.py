"""
V0.9: 隣接リージョンの階層統合 (優先度付きキュー + lazy invalidation, numpy のみ)。

色・テクスチャ・境界・SAM・サイズの正規化指標で merge_cost を計算し、最小コストの
隣接ペアから統合して 1 つの root までの完全二分マージツリーを作る。葉=1..leaf_count、
親=leaf_count+1.. を順に割り当てる。node 配列 index = node_id - 1。

  merge_cost = wc*color + wt*texture + wb*boundary + ws*sam_disagree + wsz*size
"""

from __future__ import annotations

import heapq

import numpy as np

from partition_backend.region_features import LeafFeatures, texture_vector
from partition_backend.region_graph import RegionGraph
from partition_backend.sam_guidance import SamSignatures

__all__ = ["MergeWeights", "HierarchyResult", "build_hierarchy"]

# SAM ガイドで「高信頼」とみなす guidance_score のしきい値
SAM_HIGH = 0.15


class MergeWeights:
    __slots__ = ("color", "texture", "boundary", "sam", "size")

    def __init__(self, color=0.30, texture=0.10, boundary=0.30, sam=0.25, size=0.05):
        self.color = float(color)
        self.texture = float(texture)
        self.boundary = float(boundary)
        self.sam = float(sam)
        self.size = float(size)


class HierarchyResult:
    """build_hierarchy の出力 (partition_npz.build_partition_arrays へ渡せる)。"""

    __slots__ = (
        "leaf_count", "node_count", "root_id",
        "node_left", "node_right", "node_parent", "node_area", "node_bbox",
        "node_centroid", "node_mean_lab", "node_texture", "node_merge_cost",
        "node_level",
    )

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw[k])


class _Node:
    __slots__ = ("area", "lab", "std_lum", "mean_grad", "bbox",
                 "cxsum", "cysum", "sam_id", "sam_score", "version")

    def __init__(self, area, lab, std_lum, mean_grad, bbox, cxsum, cysum,
                 sam_id, sam_score):
        self.area = area
        self.lab = lab            # np.array (3,)
        self.std_lum = std_lum
        self.mean_grad = mean_grad
        self.bbox = bbox          # [x0,y0,x1,y1] (x1,y1 排他)
        self.cxsum = cxsum
        self.cysum = cysum
        self.sam_id = sam_id
        self.sam_score = sam_score
        self.version = 0


def _dominant_sam(sig: SamSignatures, leaf_id: int) -> tuple[int, float]:
    ids, _, scores = sig.for_leaf(leaf_id)
    if ids.size == 0:
        return 0, 0.0
    j = int(np.argmax(scores))
    return int(ids[j]), float(scores[j])


def build_hierarchy(
    feat: LeafFeatures,
    graph: RegionGraph,
    sig: SamSignatures,
    weights: MergeWeights,
    image_area: int,
) -> HierarchyResult:
    leaf_count = int(feat.k)
    node_count = 2 * leaf_count - 1
    image_area = max(1, int(image_area))

    tex = texture_vector(feat)  # (K+1,4): [std_lum, mean_grad, std_a, std_b]

    # node 配列 (index = node_id - 1)
    nl = np.zeros(node_count, dtype=np.int64)
    nr = np.zeros(node_count, dtype=np.int64)
    npar = np.zeros(node_count, dtype=np.int64)
    narea = np.zeros(node_count, dtype=np.int64)
    nbbox = np.zeros((node_count, 4), dtype=np.int64)
    ncent = np.zeros((node_count, 2), dtype=np.float32)
    nlab = np.zeros((node_count, 3), dtype=np.float32)
    ntex = np.zeros((node_count, 4), dtype=np.float32)
    ncost = np.zeros(node_count, dtype=np.float32)
    nlevel = np.zeros(node_count, dtype=np.int64)

    nodes: dict[int, _Node] = {}
    for lid in range(1, leaf_count + 1):
        x, y, w, h = (int(v) for v in feat.bbox[lid])
        sam_id, sam_score = _dominant_sam(sig, lid)
        nd = _Node(
            area=int(feat.area[lid]),
            lab=feat.mean_lab[lid].astype(np.float64).copy(),
            std_lum=float(feat.std_lum[lid]),
            mean_grad=float(feat.mean_grad[lid]),
            bbox=[x, y, x + max(w, 1), y + max(h, 1)],
            cxsum=float(feat.centroid[lid, 0]) * int(feat.area[lid]),
            cysum=float(feat.centroid[lid, 1]) * int(feat.area[lid]),
            sam_id=sam_id, sam_score=sam_score,
        )
        nodes[lid] = nd
        # 葉 node 配列
        narea[lid - 1] = nd.area
        nbbox[lid - 1] = [x, y, max(w, 1), max(h, 1)]
        ncent[lid - 1] = feat.centroid[lid]
        nlab[lid - 1] = feat.mean_lab[lid]
        ntex[lid - 1] = tex[lid]
        nlevel[lid - 1] = 0

    # 隣接 (active node 間): node -> {neighbor: [shared_length, grad_sum]}
    adj: dict[int, dict[int, list]] = {i: {} for i in range(1, leaf_count + 1)}
    for a, b, length, mgrad in graph.edges():
        adj[a][b] = [int(length), float(mgrad) * int(length)]
        adj[b][a] = [int(length), float(mgrad) * int(length)]

    # 正規化定数 (初期エッジから決定的に算出)
    color_vals, tex_vals, grad_vals = [], [], []
    for a, b, length, mgrad in graph.edges():
        color_vals.append(float(np.linalg.norm(nodes[a].lab - nodes[b].lab)))
        tv = ((nodes[a].std_lum - nodes[b].std_lum) ** 2
              + (nodes[a].mean_grad - nodes[b].mean_grad) ** 2) ** 0.5
        tex_vals.append(tv)
        grad_vals.append(float(mgrad))
    color_norm = max(color_vals) if color_vals else 1.0
    tex_norm = max(tex_vals) if tex_vals else 1.0
    grad_norm = max(grad_vals) if grad_vals else 1.0
    color_norm = color_norm or 1.0
    tex_norm = tex_norm or 1.0
    grad_norm = grad_norm or 1.0

    def merge_cost(a: int, b: int) -> float:
        na, nb = nodes[a], nodes[b]
        color = float(np.linalg.norm(na.lab - nb.lab)) / color_norm
        texd = (((na.std_lum - nb.std_lum) ** 2
                 + (na.mean_grad - nb.mean_grad) ** 2) ** 0.5) / tex_norm
        pair = adj[a].get(b)
        if pair and pair[0] > 0:
            boundary = (pair[1] / pair[0]) / grad_norm
        else:
            boundary = 0.0
        sam = _sam_disagreement(na, nb)
        size = min(na.area, nb.area) / image_area
        cost = (weights.color * min(color, 1.0)
                + weights.texture * min(texd, 1.0)
                + weights.boundary * min(boundary, 1.0)
                + weights.sam * sam
                + weights.size * min(size, 1.0))
        return float(cost)

    heap: list = []

    def push(a: int, b: int):
        lo, hi = (a, b) if a < b else (b, a)
        heapq.heappush(heap, (merge_cost(lo, hi), nodes[lo].version,
                              nodes[hi].version, lo, hi))

    for a in adj:
        for b in adj[a]:
            if a < b:
                push(a, b)

    active = set(range(1, leaf_count + 1))
    next_id = leaf_count + 1

    while len(active) > 1:
        if heap:
            cost, va, vb, a, b = heapq.heappop(heap)
            if a not in active or b not in active:
                continue
            if nodes[a].version != va or nodes[b].version != vb:
                continue
        else:
            # RAG が非連結: 残りを決定的に強制統合 (cost 大)
            rem = sorted(active)
            a, b = rem[0], rem[1]
            cost = 1.0
            adj.setdefault(a, {})
            adj.setdefault(b, {})

        c = next_id
        next_id += 1
        na, nb = nodes[a], nodes[b]
        area_c = na.area + nb.area
        lab_c = (na.lab * na.area + nb.lab * nb.area) / max(area_c, 1)
        std_c = (na.std_lum * na.area + nb.std_lum * nb.area) / max(area_c, 1)
        grad_c = (na.mean_grad * na.area + nb.mean_grad * nb.area) / max(area_c, 1)
        bbox_c = [min(na.bbox[0], nb.bbox[0]), min(na.bbox[1], nb.bbox[1]),
                  max(na.bbox[2], nb.bbox[2]), max(na.bbox[3], nb.bbox[3])]
        if na.sam_score >= nb.sam_score:
            sam_id_c, sam_score_c = na.sam_id, na.sam_score
        else:
            sam_id_c, sam_score_c = nb.sam_id, nb.sam_score
        nd = _Node(area_c, lab_c, std_c, grad_c, bbox_c,
                   na.cxsum + nb.cxsum, na.cysum + nb.cysum, sam_id_c, sam_score_c)
        nodes[c] = nd

        # node 配列へ記録
        nl[c - 1] = a
        nr[c - 1] = b
        npar[a - 1] = c
        npar[b - 1] = c
        narea[c - 1] = area_c
        nbbox[c - 1] = [bbox_c[0], bbox_c[1],
                        bbox_c[2] - bbox_c[0], bbox_c[3] - bbox_c[1]]
        ncent[c - 1] = [nd.cxsum / max(area_c, 1), nd.cysum / max(area_c, 1)]
        nlab[c - 1] = lab_c.astype(np.float32)
        # texture: std_lum, mean_grad は更新、std_a/std_b は面積加重
        ta = ntex[a - 1]; tb = ntex[b - 1]
        ntex[c - 1] = [std_c, grad_c,
                       (ta[2] * na.area + tb[2] * nb.area) / max(area_c, 1),
                       (ta[3] * na.area + tb[3] * nb.area) / max(area_c, 1)]
        ncost[c - 1] = cost
        nlevel[c - 1] = max(nlevel[a - 1], nlevel[b - 1]) + 1

        # 隣接更新
        neigh_c: dict[int, list] = {}
        for old in (a, b):
            for n, val in adj.get(old, {}).items():
                if n in (a, b):
                    continue
                if n in neigh_c:
                    neigh_c[n][0] += val[0]
                    neigh_c[n][1] += val[1]
                else:
                    neigh_c[n] = [val[0], val[1]]
        # active 集合更新
        active.discard(a)
        active.discard(b)
        for old in (a, b):
            for n in list(adj.get(old, {}).keys()):
                adj.get(n, {}).pop(old, None)
            adj.pop(old, None)
        adj[c] = neigh_c
        for n, val in neigh_c.items():
            adj[n][c] = list(val)
            nodes[n].version += 1
        active.add(c)
        nd.version = 0

        for n in neigh_c:
            push(c, n)

    root_id = next_id - 1 if leaf_count > 1 else 1
    # root の parent は 0 (初期値のまま)

    return HierarchyResult(
        leaf_count=leaf_count, node_count=node_count, root_id=root_id,
        node_left=nl, node_right=nr, node_parent=npar, node_area=narea,
        node_bbox=nbbox, node_centroid=ncent, node_mean_lab=nlab,
        node_texture=ntex, node_merge_cost=ncost, node_level=nlevel,
    )


def _sam_disagreement(na: _Node, nb: _Node) -> float:
    """SAM ガイドの不一致度 (0=統合しやすい, 1=統合しにくい, 0.5=中立)。"""
    a_high = na.sam_id != 0 and na.sam_score >= SAM_HIGH
    b_high = nb.sam_id != 0 and nb.sam_score >= SAM_HIGH
    if a_high and b_high:
        return 0.0 if na.sam_id == nb.sam_id else 1.0
    return 0.5
