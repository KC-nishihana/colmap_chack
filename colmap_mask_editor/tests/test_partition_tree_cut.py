"""V0.9: 階層カット・局所 split・collapse・leaf->visible。"""

import numpy as np

from ai import partition_tree
from ai.partition_tree import (
    PartitionTree,
    cut_tree_to_count,
    split_visible_node_tree,
    collapse_to_parent,
    leaf_to_visible_node,
)

from tests._partition_helpers import simple_three_leaf


def _tree():
    return PartitionTree.from_npz(simple_three_leaf())


def test_cut_to_count_root():
    t = _tree()
    assert cut_tree_to_count(t, 1) == [t.root_id]


def test_cut_to_count_two():
    t = _tree()
    vis = cut_tree_to_count(t, 2)
    assert len(vis) == 2
    # root(5) を分割 -> 子 4 と 3
    assert set(vis) == {4, 3}


def test_cut_to_count_all_leaves():
    t = _tree()
    vis = cut_tree_to_count(t, 3)
    assert set(vis) == {1, 2, 3}


def test_cut_caps_at_leaf_count():
    t = _tree()
    vis = cut_tree_to_count(t, 99)
    assert set(vis) == {1, 2, 3}  # 葉以上には分割できない


def test_split_visible_node():
    t = _tree()
    vis = [4, 3]
    out = split_visible_node_tree(t, vis, 4)
    assert set(out) == {1, 2, 3}


def test_split_leaf_noop():
    t = _tree()
    vis = [1, 2, 3]
    out = split_visible_node_tree(t, vis, 1)
    assert set(out) == {1, 2, 3}


def test_collapse_to_parent():
    t = _tree()
    vis = [1, 2, 3]
    out = collapse_to_parent(t, vis, 1)  # 1 と兄弟 2 -> 親 4
    assert set(out) == {4, 3}


def test_leaf_to_visible_node():
    t = _tree()
    parent = simple_three_leaf()["node_parent"]
    # 表示 {4,3}: 葉1 -> 祖先 4
    assert leaf_to_visible_node(1, {4, 3}, parent) == 4
    assert leaf_to_visible_node(2, {4, 3}, parent) == 4
    assert leaf_to_visible_node(3, {4, 3}, parent) == 3
    # 表示 {5}=root: どの葉も 5
    assert leaf_to_visible_node(1, {5}, parent) == 5


def test_ancestor_in_set():
    t = _tree()
    assert t.ancestor_in_set(1, {4, 3}) == 4
    assert t.ancestor_in_set(3, {1, 2, 3}) == 3


def test_leaves_under():
    t = _tree()
    assert sorted(t.leaves_under(5)) == [1, 2, 3]
    assert sorted(t.leaves_under(4)) == [1, 2]
    assert t.leaves_under(3) == [3]
