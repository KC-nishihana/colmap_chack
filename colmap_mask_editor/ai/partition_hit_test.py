"""
V0.9: クリック位置 -> 葉 region_id -> 表示階層の祖先ノードを高速取得 (numpy のみ)。

partition.npz の run-length データ (C-order) から、クリック画素の葉 region_id を
二分探索で求める。全ラベルマップを毎回復号しない。求めた葉から現在の表示集合の
祖先ノードを返す。

  flat_index = y * width + x
"""

from __future__ import annotations

import numpy as np

from ai import partition_rle
from ai.partition_tree import leaf_to_visible_node

__all__ = ["PartitionHitTester"]


class PartitionHitTester:
    """run-length region map の累積和をキャッシュしてクリック判定を高速化する。"""

    __slots__ = ("_ids", "_lengths", "_cum", "_h", "_w", "_parent")

    def __init__(self, partition_data: dict) -> None:
        self._ids = np.asarray(partition_data["run_region_ids"])
        self._lengths = np.asarray(partition_data["run_lengths"])
        shape = np.asarray(partition_data["image_shape"])
        self._h, self._w = int(shape[0]), int(shape[1])
        self._parent = np.asarray(partition_data["node_parent"], dtype=np.int64)
        # 累積和は 1 度だけ計算して再利用 (クリックごとに復号しない)
        self._cum = np.cumsum(self._lengths.astype(np.int64))

    @property
    def shape(self) -> tuple[int, int]:
        return (self._h, self._w)

    def leaf_at(self, x: int, y: int) -> int | None:
        """(x, y) の葉 region_id を返す。範囲外は None。"""
        xi, yi = int(x), int(y)
        if xi < 0 or yi < 0 or xi >= self._w or yi >= self._h:
            return None
        return partition_rle.region_at_point(
            self._ids, self._lengths, self._w, xi, yi, cum=self._cum
        )

    def node_at(self, x: int, y: int, visible_nodes) -> int | None:
        """(x, y) の葉から、表示集合 visible_nodes の祖先ノードを返す。"""
        leaf = self.leaf_at(x, y)
        if leaf is None:
            return None
        return leaf_to_visible_node(leaf, visible_nodes, self._parent)
