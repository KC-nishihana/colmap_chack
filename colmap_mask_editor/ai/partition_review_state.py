"""
V0.9: 階層ノードの判断状態 (keep / remove / unreviewed) と継承解決 (numpy のみ)。

判断はノード単位で partition_review.json["node_decisions"] に保存する
(明示判断のみ。unreviewed はキー不在で表す)。partition.npz は書き換えない。

継承: 葉の実効判断は、葉から root 方向へ遡り最初に見つかった明示判断。
子で明示判断すれば親判断を上書きできる。
"""

from __future__ import annotations

from enum import Enum

import numpy as np

__all__ = [
    "RegionDecision",
    "VALID_DECISIONS",
    "normalize_node_decisions",
    "effective_decision",
    "effective_leaf_decisions",
    "set_node_decision",
    "set_parent_decision_clearing_descendants",
    "descendant_explicit_nodes",
    "pixel_rates",
    "is_complete",
    "unreviewed_leaf_ids",
]


class RegionDecision(Enum):
    UNREVIEWED = "unreviewed"
    KEEP = "keep"
    REMOVE = "remove"

    @classmethod
    def from_str(cls, value: str) -> "RegionDecision":
        try:
            return cls(value)
        except ValueError as e:
            raise ValueError(f"不明な判断状態: {value!r}") from e


VALID_DECISIONS = frozenset(d.value for d in RegionDecision)
# 明示判断のみ (unreviewed は保存しない)
_EXPLICIT = frozenset({RegionDecision.KEEP.value, RegionDecision.REMOVE.value})


def normalize_node_decisions(decisions: dict, node_count: int | None = None) -> dict[str, str]:
    """
    node_decisions を検証・正規化する。明示判断 (keep/remove) のみ残す。

    unreviewed は保存しない (キー削除)。範囲外 node_id は node_count 指定時に除外。
    """
    out: dict[str, str] = {}
    for key, value in (decisions or {}).items():
        skey = str(int(key))
        if value == RegionDecision.UNREVIEWED.value:
            continue
        if value not in _EXPLICIT:
            raise ValueError(f"node {skey}: 不明な判断状態 {value!r}")
        if node_count is not None:
            nid = int(skey)
            if nid < 1 or nid > int(node_count):
                continue
        out[skey] = value
    return out


def effective_decision(parent_array, node_id: int, decisions: dict) -> str:
    """
    node_id から root 方向へ遡り、最初の明示判断を返す。無ければ unreviewed。

    parent_array は index = node_id - 1 の親配列 (root の親 = 0)。
    """
    parent = np.asarray(parent_array, dtype=np.int64)
    cur = int(node_id)
    while cur != 0:
        v = decisions.get(str(cur))
        if v in _EXPLICIT:
            return v
        cur = int(parent[cur - 1])
    return RegionDecision.UNREVIEWED.value


def effective_leaf_decisions(parent_array, leaf_count: int, decisions: dict) -> np.ndarray:
    """
    各葉 (1..leaf_count) の実効判断を解決した配列 (index 0 未使用)。

    値: 0=unreviewed, 1=keep, 2=remove。継承を上方向探索で解決。
    """
    parent = np.asarray(parent_array, dtype=np.int64)
    code = {RegionDecision.KEEP.value: 1, RegionDecision.REMOVE.value: 2}
    out = np.zeros(int(leaf_count) + 1, dtype=np.uint8)
    # ノードごとの明示判断コードを配列化
    for leaf in range(1, int(leaf_count) + 1):
        cur = leaf
        val = 0
        while cur != 0:
            v = decisions.get(str(cur))
            if v in _EXPLICIT:
                val = code[v]
                break
            cur = int(parent[cur - 1])
        out[leaf] = val
    return out


def descendant_explicit_nodes(left, right, node_id: int, decisions: dict) -> list[int]:
    """node_id の子孫 (自身は含まない) で明示判断を持つノード id を列挙する。"""
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    found: list[int] = []
    l = int(left[node_id - 1])
    r = int(right[node_id - 1])
    stack = [x for x in (l, r) if x != 0]
    while stack:
        n = stack.pop()
        if str(n) in decisions and decisions[str(n)] in _EXPLICIT:
            found.append(n)
        cl = int(left[n - 1])
        cr = int(right[n - 1])
        if cl:
            stack.append(cl)
        if cr:
            stack.append(cr)
    return sorted(found)


def set_node_decision(decisions: dict, node_id: int, decision: RegionDecision) -> dict[str, str]:
    """1 ノードの判断を更新した新しい dict を返す。UNREVIEWED はキー削除。"""
    out = dict(decisions)
    key = str(int(node_id))
    if decision == RegionDecision.UNREVIEWED:
        out.pop(key, None)
    else:
        out[key] = decision.value
    return out


def set_parent_decision_clearing_descendants(
    left, right, decisions: dict, node_id: int, decision: RegionDecision,
) -> dict[str, str]:
    """
    親ノードへ判断を設定し、子孫の明示判断を削除する (承認済み前提)。

    呼び出し側で先に descendant_explicit_nodes を確認しユーザー確認すること。
    """
    out = dict(decisions)
    for n in descendant_explicit_nodes(left, right, node_id, decisions):
        out.pop(str(n), None)
    return set_node_decision(out, node_id, decision)


def pixel_rates(parent_array, leaf_count: int, leaf_areas, decisions: dict) -> dict:
    """
    葉面積から KEEP / REMOVE / 未確認 の画素率を計算する。

    leaf_areas は partition.npz の node_area (index = node_id - 1)。葉 1..K は
    index 0..K-1 を使う。全解像度マスクを生成しない。
    """
    eff = effective_leaf_decisions(parent_array, leaf_count, decisions)
    k = int(leaf_count)
    # node_area は index = node_id - 1。葉 1..k は index 0..k-1。
    areas = np.asarray(leaf_areas, dtype=np.int64)[:k]
    effk = eff[1:k + 1]
    total = int(areas.sum())
    total_safe = total if total > 0 else 1
    keep = int(areas[effk == 1].sum())
    remove = int(areas[effk == 2].sum())
    unrev = int(areas[effk == 0].sum())
    return {
        "total_pixels": total,
        "keep_pixels": keep,
        "remove_pixels": remove,
        "unreviewed_pixels": unrev,
        "keep_ratio": keep / total_safe,
        "remove_ratio": remove / total_safe,
        "unreviewed_ratio": unrev / total_safe,
        "assigned_ratio": (keep + remove + unrev) / total_safe,
    }


def unreviewed_leaf_ids(parent_array, leaf_count: int, decisions: dict) -> list[int]:
    """実効判断が unreviewed の葉 region_id リスト。"""
    eff = effective_leaf_decisions(parent_array, leaf_count, decisions)
    return [int(i) for i in range(1, int(leaf_count) + 1) if eff[i] == 0]


def is_complete(parent_array, leaf_count: int, decisions: dict) -> bool:
    """全葉に実効判断 (keep/remove) があれば True。"""
    eff = effective_leaf_decisions(parent_array, leaf_count, decisions)
    return bool(np.all(eff[1:int(leaf_count) + 1] != 0))
