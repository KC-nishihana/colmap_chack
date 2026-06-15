"""
V0.8: レビュー画面のクリック候補判定と復号LRUキャッシュ (numpy のみ・Qt非依存)。

クリック位置の候補抽出は次の順序:
  1. bbox にクリック座標が含まれる候補を抽出 (安価)
  2. rle_contains_point() で正確に判定 (全マスクを復号しない)
  3. 面積昇順で並べる
  4. 最も小さい候補を初期選択。重複候補は Tab で切替

復号は現在レビュー中の画像だけを対象に LRU キャッシュする (8K で多数同時復号を避ける)。
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from ai import amg_rle

__all__ = ["candidates_at_point", "cycle_index", "MaskDecodeCache"]


def candidates_at_point(npz_data, x: int, y: int) -> list[int]:
    """
    (x, y) を含む segment の index リストを面積昇順 (小さい順) で返す。

    npz_data: amg_npz.load_segments_npz / np.load の結果 (dict 風)。
    返すのは segment の配列 index (0..N-1)。空ならクリックは前景無し。
    """
    image_shape = np.asarray(npz_data["image_shape"])
    h, w = int(image_shape[0]), int(image_shape[1])
    xi, yi = int(x), int(y)
    if xi < 0 or yi < 0 or xi >= w or yi >= h:
        return []

    bbox = np.asarray(npz_data["bbox_xywh"])
    area = np.asarray(npz_data["area"])
    n = bbox.shape[0]

    hits: list[tuple[int, int]] = []  # (area, index)
    for i in range(n):
        bx, by, bw, bh = (int(v) for v in bbox[i])
        if xi < bx or yi < by or xi >= bx + bw or yi >= by + bh:
            continue  # bbox 外 -> 安価に除外
        counts = amg_rle.unpack_counts(npz_data, i)
        if amg_rle.rle_contains_point(counts, h, w, xi, yi):
            hits.append((int(area[i]), i))

    hits.sort(key=lambda t: (t[0], t[1]))  # 面積昇順 (同面積は index 安定)
    return [i for _, i in hits]


def cycle_index(candidates: list[int], current: int | None, forward: bool = True) -> int | None:
    """
    候補リスト内で current の次/前へ切り替える (Tab / Shift+Tab)。

    current が候補外/None のときは先頭 (forward) または末尾 (backward) を返す。
    """
    if not candidates:
        return None
    if current is None or current not in candidates:
        return candidates[0] if forward else candidates[-1]
    pos = candidates.index(current)
    pos = (pos + 1) % len(candidates) if forward else (pos - 1) % len(candidates)
    return candidates[pos]


class MaskDecodeCache:
    """現在画像のセグメント復号マスクの LRU キャッシュ (画像切替時に破棄)。"""

    def __init__(self, npz_data, max_size: int = 12) -> None:
        self._data = npz_data
        image_shape = np.asarray(npz_data["image_shape"])
        self._h, self._w = int(image_shape[0]), int(image_shape[1])
        self._max = max(1, int(max_size))
        self._cache: "OrderedDict[int, np.ndarray]" = OrderedDict()

    @property
    def shape(self) -> tuple[int, int]:
        return (self._h, self._w)

    def get(self, segment_index: int) -> np.ndarray:
        """segment_index の (h,w) uint8(0/255) マスクを返す (LRU)。"""
        i = int(segment_index)
        if i in self._cache:
            self._cache.move_to_end(i)
            return self._cache[i]
        counts = amg_rle.unpack_counts(self._data, i)
        mask = amg_rle.decode_rle(counts, self._h, self._w)
        self._cache[i] = mask
        self._cache.move_to_end(i)
        while len(self._cache) > self._max:
            self._cache.popitem(last=False)
        return mask

    def union(self, segment_indices) -> np.ndarray:
        """指定 segment の和集合 (bool, h,w)。判断変更時に必要分だけ再計算する。"""
        out = np.zeros((self._h, self._w), dtype=bool)
        for i in segment_indices:
            out |= self.get(i) > 0
        return out

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)
