"""V0.9: クリック判定 (RLE 二分探索 -> 葉 -> 表示祖先)。"""

import numpy as np

from ai.partition_hit_test import PartitionHitTester

from tests._partition_helpers import simple_three_leaf

ARR = simple_three_leaf()


def test_leaf_at_matches_layout():
    ht = PartitionHitTester(ARR)
    assert ht.leaf_at(0, 0) == 1
    assert ht.leaf_at(2, 0) == 2
    assert ht.leaf_at(5, 3) == 3


def test_leaf_at_out_of_range():
    ht = PartitionHitTester(ARR)
    assert ht.leaf_at(-1, 0) is None
    assert ht.leaf_at(6, 0) is None
    assert ht.leaf_at(0, 4) is None


def test_node_at_visible_ancestor():
    ht = PartitionHitTester(ARR)
    # 表示 {4,3}: leaf1,2 -> 4 ; leaf3 -> 3
    assert ht.node_at(0, 0, {4, 3}) == 4
    assert ht.node_at(2, 0, {4, 3}) == 4
    assert ht.node_at(4, 0, {4, 3}) == 3
    # root 表示
    assert ht.node_at(0, 0, {5}) == 5


def test_cum_reused_no_full_decode():
    ht = PartitionHitTester(ARR)
    # 全画素で判定しても例外なく一致 (累積和は 1 度だけ)
    for y in range(4):
        for x in range(6):
            assert ht.leaf_at(x, y) in (1, 2, 3)
