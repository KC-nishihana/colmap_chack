"""
V0.8: セグメント判断状態 (keep / remove / unreviewed) の表現と検証。

判断状態は manifest.json["review"]["decisions"] に小さな JSON として保存し、
NPZ は一切書き換えない。
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class SegmentDecision(Enum):
    UNREVIEWED = "unreviewed"
    KEEP = "keep"
    REMOVE = "remove"

    @classmethod
    def from_str(cls, value: str) -> "SegmentDecision":
        try:
            return cls(value)
        except ValueError as e:
            raise ValueError(f"不明な判断状態: {value!r}") from e


VALID_DECISIONS = frozenset(d.value for d in SegmentDecision)

__all__ = [
    "SegmentDecision",
    "VALID_DECISIONS",
    "normalize_decisions",
    "default_decisions",
    "set_decision",
    "count_decisions",
    "is_review_complete",
]


def default_decisions(segment_ids) -> dict[str, str]:
    """全 segment を unreviewed で初期化した decisions を作る (キーは文字列)。"""
    return {str(int(sid)): SegmentDecision.UNREVIEWED.value for sid in segment_ids}


def normalize_decisions(decisions: dict[str, Any], segment_ids=None) -> dict[str, str]:
    """
    decisions を検証・正規化する。不明な値は拒否 (ValueError)。

    segment_ids を渡した場合、存在しない segment への decision は除外し、
    欠けている segment は unreviewed で補完する。
    """
    out: dict[str, str] = {}
    valid_ids = None if segment_ids is None else {str(int(s)) for s in segment_ids}
    for key, value in (decisions or {}).items():
        skey = str(key)
        if value not in VALID_DECISIONS:
            raise ValueError(f"segment {skey}: 不明な判断状態 {value!r}")
        if valid_ids is not None and skey not in valid_ids:
            continue  # 存在しない segment_id の decision は捨てる
        out[skey] = value
    if valid_ids is not None:
        for skey in valid_ids:
            out.setdefault(skey, SegmentDecision.UNREVIEWED.value)
    return out


def set_decision(decisions: dict[str, str], segment_id: int, decision: SegmentDecision) -> dict[str, str]:
    """decisions の 1 件を更新した新しい dict を返す (元を破壊しない)。"""
    updated = dict(decisions)
    updated[str(int(segment_id))] = decision.value
    return updated


def count_decisions(decisions: dict[str, str]) -> dict[str, int]:
    """keep / remove / unreviewed の件数を集計する。"""
    counts = {d.value: 0 for d in SegmentDecision}
    for value in decisions.values():
        if value in counts:
            counts[value] += 1
    return counts


def is_review_complete(decisions: dict[str, str]) -> bool:
    """未確認 (unreviewed) が 1 件も無ければ True。"""
    return all(v != SegmentDecision.UNREVIEWED.value for v in decisions.values())
