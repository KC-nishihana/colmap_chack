"""V0.8: 判断状態 (keep/remove/unreviewed) の検証テスト。"""

import pytest

from ai.amg_review_state import (
    SegmentDecision,
    count_decisions,
    default_decisions,
    is_review_complete,
    normalize_decisions,
    set_decision,
)


def test_default_all_unreviewed():
    d = default_decisions([1, 2, 3])
    assert d == {"1": "unreviewed", "2": "unreviewed", "3": "unreviewed"}


def test_normalize_rejects_unknown():
    with pytest.raises(ValueError):
        normalize_decisions({"1": "maybe"})


def test_normalize_fills_and_drops(tmp_path):
    out = normalize_decisions({"1": "keep", "9": "remove"}, segment_ids=[1, 2])
    assert out["1"] == "keep"
    assert out["2"] == "unreviewed"   # 補完
    assert "9" not in out             # 存在しない segment を捨てる


def test_set_decision_immutable():
    d = default_decisions([1, 2])
    d2 = set_decision(d, 1, SegmentDecision.KEEP)
    assert d["1"] == "unreviewed"  # 元は不変
    assert d2["1"] == "keep"


def test_count_and_complete():
    d = {"1": "keep", "2": "remove", "3": "unreviewed"}
    counts = count_decisions(d)
    assert counts == {"unreviewed": 1, "keep": 1, "remove": 1}
    assert is_review_complete(d) is False
    assert is_review_complete({"1": "keep", "2": "remove"}) is True


def test_from_str():
    assert SegmentDecision.from_str("keep") is SegmentDecision.KEEP
    with pytest.raises(ValueError):
        SegmentDecision.from_str("x")
