"""
V0.10: SAM が同一対象に対して生成した重複候補をグループ化する (numpy のみ)。

目的: REMOVE_ONLY レビューで、ほぼ同じ領域を指す複数候補を 1 件の代表候補へ
まとめて表示数を減らす。候補を NPZ から削除せず、表示抑制のための索引だけを作る。

同一グループの条件 (どちらかを満たせば同一グループ):
  - IoU >= group_iou_threshold              (既定 0.85)
  - 小さい候補の containment >= group_containment_threshold が大きい候補に対し成立
                                            (既定 0.95)

最適化:
  - bbox が交差しない候補同士は RLE 比較しない
  - RLE 比較は dense マスクへ復号しない (amg_rle_overlap を使用)

代表候補の選択 (グループ内で決定的):
  1. predicted_iou * stability_score が高い
  2. 同程度 (同値) なら面積が大きい
  3. 同値なら segment_id が小さい

group_id は「グループ内の最小 segment_index」で昇順に並べて 0..G-1 を割り当てる。
同じ入力に対しては常に同じ group_id になる (NPZ の並びが決定的なため)。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ai import amg_rle, amg_rle_overlap as ov

__all__ = [
    "GroupingResult",
    "bbox_intersects",
    "quality_scores",
    "edge_touch_flags",
    "group_candidates",
]


@dataclass(frozen=True)
class GroupingResult:
    """候補グループ化の結果 (すべて segment index 順 0..N-1 に整列)。"""
    group_ids: np.ndarray                 # uint32 (N,)  各 segment の group_id
    representative_segment_ids: np.ndarray  # uint32 (N,)  所属グループの代表 segment_id
    is_representative: np.ndarray         # uint8 (N,)   その segment が代表なら 1
    group_count: int

    def representative_indices(self) -> set[int]:
        """代表候補の segment index 集合を返す。"""
        return {int(i) for i in np.flatnonzero(self.is_representative)}


def bbox_intersects(b1, b2) -> bool:
    """xywh bbox が交差 (面積 > 0 の重なり) するか。接するだけ (隣接) は非交差扱い。"""
    x1, y1, w1, h1 = (int(v) for v in b1)
    x2, y2, w2, h2 = (int(v) for v in b2)
    if x1 + w1 <= x2 or x2 + w2 <= x1:
        return False
    if y1 + h1 <= y2 or y2 + h2 <= y1:
        return False
    return True


def quality_scores(npz_data) -> np.ndarray:
    """quality = predicted_iou * stability_score を float32 (N,) で返す。"""
    iou = np.asarray(npz_data["predicted_iou"], dtype=np.float64)
    stab = np.asarray(npz_data["stability_score"], dtype=np.float64)
    return (iou * stab).astype(np.float32)


def edge_touch_flags(npz_data) -> np.ndarray:
    """bbox が画像端に接する候補を 1、それ以外を 0 とする uint8 (N,)。"""
    image_shape = np.asarray(npz_data["image_shape"])
    h, w = int(image_shape[0]), int(image_shape[1])
    bbox = np.asarray(npz_data["bbox_xywh"])
    n = bbox.shape[0]
    flags = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        bx, by, bw, bh = (int(v) for v in bbox[i])
        if bx <= 0 or by <= 0 or bx + bw >= w or by + bh >= h:
            flags[i] = 1
    return flags


class _UnionFind:
    def __init__(self, n: int) -> None:
        self._p = list(range(n))

    def find(self, x: int) -> int:
        root = x
        while self._p[root] != root:
            root = self._p[root]
        while self._p[x] != root:
            self._p[x], x = root, self._p[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # 小さい root を親にして決定性を保つ
            if rb < ra:
                ra, rb = rb, ra
            self._p[rb] = ra


def group_candidates(
    npz_data,
    *,
    iou_threshold: float = 0.85,
    containment_threshold: float = 0.95,
) -> GroupingResult:
    """
    候補をグループ化し、group_ids と representative_segment_ids を返す。

    bbox 非交差ペアは比較せず、交差ペアだけ RLE で IoU / containment を計算する。
    """
    segment_ids = np.asarray(npz_data["segment_ids"]).astype(np.int64)
    bbox = np.asarray(npz_data["bbox_xywh"])
    area = np.asarray(npz_data["area"]).astype(np.int64)
    n = int(segment_ids.shape[0])

    uf = _UnionFind(n)

    # bbox 交差ペアだけ RLE 比較。counts はペア計算時に必要分だけ取り出す。
    counts_cache: dict[int, np.ndarray] = {}

    def counts(i: int) -> np.ndarray:
        c = counts_cache.get(i)
        if c is None:
            c = amg_rle.unpack_counts(npz_data, i)
            counts_cache[i] = c
        return c

    for i in range(n):
        bi = bbox[i]
        for j in range(i + 1, n):
            if not bbox_intersects(bi, bbox[j]):
                continue
            ci, cj = counts(i), counts(j)
            inter = ov.rle_intersection_area(ci, cj)
            if inter == 0:
                continue
            ai, aj = int(area[i]), int(area[j])
            union = ai + aj - inter
            iou = (inter / union) if union > 0 else 0.0
            # 小さい候補が大きい候補にどれだけ含まれるか
            small_area = ai if ai <= aj else aj
            containment = (inter / small_area) if small_area > 0 else 0.0
            if iou >= iou_threshold or containment >= containment_threshold:
                uf.union(i, j)

    # group root -> メンバー index
    members: dict[int, list[int]] = {}
    for i in range(n):
        members.setdefault(uf.find(i), []).append(i)

    # group_id は「グループ内最小 segment index」昇順で 0..G-1
    roots_sorted = sorted(members.keys(), key=lambda r: min(members[r]))
    group_id_of_root = {r: gid for gid, r in enumerate(roots_sorted)}

    quality = quality_scores(npz_data)

    group_ids = np.zeros(n, dtype=np.uint32)
    representative_segment_ids = np.zeros(n, dtype=np.uint32)
    is_representative = np.zeros(n, dtype=np.uint8)

    for r, idxs in members.items():
        gid = group_id_of_root[r]
        # 代表候補: quality desc -> area desc -> segment_id asc
        rep = min(
            idxs,
            key=lambda i: (-float(quality[i]), -int(area[i]), int(segment_ids[i])),
        )
        rep_sid = int(segment_ids[rep])
        for i in idxs:
            group_ids[i] = gid
            representative_segment_ids[i] = rep_sid
        is_representative[rep] = 1

    return GroupingResult(
        group_ids=group_ids,
        representative_segment_ids=representative_segment_ids,
        is_representative=is_representative,
        group_count=len(roots_sorted),
    )
