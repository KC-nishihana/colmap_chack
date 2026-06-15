"""V0.9: ノード判断・継承・上書き・画素率・完了判定。"""

import numpy as np
import pytest

from ai import partition_review_state as prs
from ai.partition_review_state import RegionDecision

from tests._partition_helpers import simple_three_leaf

# tree: 1,2 葉 -> 4; 4,3 -> 5(root)
ARR = simple_three_leaf()
PARENT = ARR["node_parent"]
LEFT = ARR["node_left"]
RIGHT = ARR["node_right"]
AREAS = ARR["node_area"]  # 各葉 8px


def test_inherit_from_parent():
    dec = {"5": "remove"}
    eff = prs.effective_leaf_decisions(PARENT, 3, dec)
    assert eff[1] == 2 and eff[2] == 2 and eff[3] == 2  # 全継承 remove


def test_child_overrides_parent():
    dec = {"5": "remove", "1": "keep"}
    eff = prs.effective_leaf_decisions(PARENT, 3, dec)
    assert eff[1] == 1  # keep で上書き
    assert eff[2] == 2 and eff[3] == 2


def test_descendant_explicit_nodes():
    dec = {"4": "keep", "1": "remove"}
    found = prs.descendant_explicit_nodes(LEFT, RIGHT, 4, dec)
    assert found == [1]


def test_set_parent_clears_descendants():
    dec = {"1": "keep", "2": "remove"}
    out = prs.set_parent_decision_clearing_descendants(
        LEFT, RIGHT, dec, 4, RegionDecision.REMOVE)
    assert out == {"4": "remove"}  # 子 1,2 の明示判断は削除


def test_normalize_drops_unreviewed():
    dec = {"1": "keep", "2": "unreviewed", "3": "remove"}
    out = prs.normalize_node_decisions(dec)
    assert out == {"1": "keep", "3": "remove"}


def test_pixel_rates():
    dec = {"5": "keep", "3": "remove"}
    rates = prs.pixel_rates(PARENT, 3, AREAS, dec)
    assert rates["total_pixels"] == 24
    assert rates["keep_pixels"] == 16  # leaf1,2
    assert rates["remove_pixels"] == 8  # leaf3
    assert rates["unreviewed_pixels"] == 0
    assert abs(rates["keep_ratio"] - 16 / 24) < 1e-6


def test_is_complete():
    assert not prs.is_complete(PARENT, 3, {})
    assert prs.is_complete(PARENT, 3, {"5": "keep"})
    assert not prs.is_complete(PARENT, 3, {"1": "keep"})  # 2,3 未確認


def test_unreviewed_leaf_ids():
    assert prs.unreviewed_leaf_ids(PARENT, 3, {"1": "keep"}) == [2, 3]
    assert prs.unreviewed_leaf_ids(PARENT, 3, {"5": "remove"}) == []


def test_set_node_decision_unreviewed_removes_key():
    dec = {"1": "keep"}
    out = prs.set_node_decision(dec, 1, RegionDecision.UNREVIEWED)
    assert "1" not in out


def test_normalize_rejects_unknown():
    with pytest.raises(ValueError):
        prs.normalize_node_decisions({"1": "bogus"})
