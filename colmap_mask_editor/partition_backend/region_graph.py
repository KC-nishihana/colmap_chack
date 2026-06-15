"""
V0.9: Region Adjacency Graph (RAG) の構築 (numpy のみ)。

葉リージョンの 4 近傍隣接 (上下左右) のみを隣接とみなす。斜め接触は隣接にしない。
各隣接エッジに shared_boundary_length と mean_boundary_gradient を保存する。
同じ組み合わせ (a<b) は重複登録しない。
"""

from __future__ import annotations

import numpy as np

__all__ = ["RegionGraph", "build_region_graph"]


class RegionGraph:
    """無向 RAG。エッジは (a<b) で一意。"""

    __slots__ = ("k", "edge_a", "edge_b", "shared_length", "mean_gradient", "_adj")

    def __init__(self, k, edge_a, edge_b, shared_length, mean_gradient):
        self.k = int(k)
        self.edge_a = np.asarray(edge_a, dtype=np.int64)
        self.edge_b = np.asarray(edge_b, dtype=np.int64)
        self.shared_length = np.asarray(shared_length, dtype=np.int64)
        self.mean_gradient = np.asarray(mean_gradient, dtype=np.float64)
        self._adj: dict[int, set[int]] | None = None

    @property
    def num_edges(self) -> int:
        return int(self.edge_a.shape[0])

    def adjacency(self) -> dict[int, set[int]]:
        """region_id -> 隣接 region_id 集合 (遅延構築)。"""
        if self._adj is None:
            adj: dict[int, set[int]] = {}
            for a, b in zip(self.edge_a.tolist(), self.edge_b.tolist()):
                adj.setdefault(a, set()).add(b)
                adj.setdefault(b, set()).add(a)
            self._adj = adj
        return self._adj

    def edges(self):
        """(a, b, shared_length, mean_gradient) のイテレータ。"""
        for i in range(self.num_edges):
            yield (int(self.edge_a[i]), int(self.edge_b[i]),
                   int(self.shared_length[i]), float(self.mean_gradient[i]))


def _collect(a, b, g):
    diff = a != b
    return a[diff].astype(np.int64), b[diff].astype(np.int64), g[diff].astype(np.float64)


def build_region_graph(labels: np.ndarray, grad: np.ndarray, k: int | None = None) -> RegionGraph:
    """
    葉ラベル (1..K) と勾配画像から RAG を構築する。

    境界の勾配強度は隣り合う 2 画素の勾配平均を用いる。
    """
    arr = np.asarray(labels)
    if k is None:
        k = int(arr.max())
    grad = np.asarray(grad).astype(np.float64)

    # 水平隣接
    ha, hb, hg = _collect(
        arr[:, :-1].reshape(-1), arr[:, 1:].reshape(-1),
        0.5 * (grad[:, :-1].reshape(-1) + grad[:, 1:].reshape(-1)),
    )
    # 垂直隣接
    va, vb, vg = _collect(
        arr[:-1, :].reshape(-1), arr[1:, :].reshape(-1),
        0.5 * (grad[:-1, :].reshape(-1) + grad[1:, :].reshape(-1)),
    )
    a = np.concatenate([ha, va])
    b = np.concatenate([hb, vb])
    g = np.concatenate([hg, vg])

    if a.size == 0:
        return RegionGraph(k, [], [], [], [])

    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    key = lo * (k + 1) + hi
    uniq, inv = np.unique(key, return_inverse=True)
    count = np.bincount(inv).astype(np.int64)
    grad_sum = np.bincount(inv, weights=g)
    edge_a = (uniq // (k + 1)).astype(np.int64)
    edge_b = (uniq % (k + 1)).astype(np.int64)
    mean_grad = grad_sum / np.maximum(count, 1)
    return RegionGraph(k, edge_a, edge_b, count, mean_grad)
