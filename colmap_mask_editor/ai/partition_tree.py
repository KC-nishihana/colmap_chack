"""
V0.9: 階層カット・局所 split・collapse・leaf->visible のツリー操作 (numpy のみ)。

partition.npz の node 配列 (index = node_id - 1) を読み取り専用に扱う。判断状態は
別管理 (partition_review_state)。表示カットとは独立して判断ノードへ保存する。

ノード規約: 0=無効, 1..leaf_count=葉, leaf_count+1..=親, root=node_count。
葉は left=right=0, merge_cost=0。
"""

from __future__ import annotations

import heapq

import numpy as np

__all__ = [
    "PartitionTree",
    "cut_tree_to_count",
    "split_visible_node",
    "collapse_siblings",
    "leaf_to_visible_node",
]


class PartitionTree:
    """partition.npz の node 配列をラップした読み取り専用ツリー。"""

    __slots__ = ("left", "right", "parent", "merge_cost", "area",
                 "leaf_count", "node_count", "root_id")

    def __init__(self, *, node_left, node_right, node_parent, node_merge_cost,
                 node_area, leaf_count, node_count, root_id):
        self.left = np.asarray(node_left, dtype=np.int64)
        self.right = np.asarray(node_right, dtype=np.int64)
        self.parent = np.asarray(node_parent, dtype=np.int64)
        self.merge_cost = np.asarray(node_merge_cost, dtype=np.float64)
        self.area = np.asarray(node_area, dtype=np.int64)
        self.leaf_count = int(leaf_count)
        self.node_count = int(node_count)
        self.root_id = int(root_id)

    @classmethod
    def from_npz(cls, data: dict) -> "PartitionTree":
        return cls(
            node_left=data["node_left"], node_right=data["node_right"],
            node_parent=data["node_parent"], node_merge_cost=data["node_merge_cost"],
            node_area=data["node_area"],
            leaf_count=int(np.asarray(data["leaf_count"])[0]),
            node_count=int(np.asarray(data["node_count"])[0]),
            root_id=int(np.asarray(data["root_id"])[0]),
        )

    # --- ノード問い合わせ --- #
    def is_leaf(self, node_id: int) -> bool:
        return int(self.left[node_id - 1]) == 0

    def children(self, node_id: int) -> tuple[int, int]:
        return int(self.left[node_id - 1]), int(self.right[node_id - 1])

    def parent_of(self, node_id: int) -> int:
        return int(self.parent[node_id - 1])

    def cost_of(self, node_id: int) -> float:
        return float(self.merge_cost[node_id - 1])

    def leaves_under(self, node_id: int) -> list[int]:
        """node_id 以下の葉 region_id を列挙する (DFS, 決定的)。"""
        out: list[int] = []
        stack = [int(node_id)]
        while stack:
            n = stack.pop()
            if self.is_leaf(n):
                out.append(n)
            else:
                l, r = self.children(n)
                stack.append(r)
                stack.append(l)
        return out

    def ancestor_in_set(self, leaf_id: int, visible: set[int]) -> int:
        """leaf_id から root 方向へ遡り、visible に含まれる最初のノードを返す。"""
        cur = int(leaf_id)
        while cur != 0:
            if cur in visible:
                return cur
            cur = self.parent_of(cur)
        return self.root_id


def cut_tree_to_count(tree: PartitionTree, target_count: int) -> list[int]:
    """
    root から開始し、表示集合内で merge_cost が最も高い内部ノードを子へ分割する。

    表示数が target_count に達するか、これ以上分割できる内部ノードが無くなるまで繰り返す。
    返り値は表示ノード id の昇順リスト。弱い統合 (高コスト) から分割される。
    """
    target = max(1, int(target_count))
    visible: set[int] = {tree.root_id}
    # 分割候補ヒープ: (-cost, node_id)
    heap: list = []

    def maybe_push(n: int):
        if not tree.is_leaf(n):
            heapq.heappush(heap, (-tree.cost_of(n), n))

    maybe_push(tree.root_id)
    while len(visible) < target and heap:
        _, n = heapq.heappop(heap)
        if n not in visible or tree.is_leaf(n):
            continue
        l, r = tree.children(n)
        visible.discard(n)
        visible.add(l)
        visible.add(r)
        maybe_push(l)
        maybe_push(r)
    return sorted(visible)


def split_visible_node(visible_nodes, node_id: int) -> list[int]:
    """
    表示集合の node_id を左右の子へ置換する (1 段階細分化)。

    visible_nodes は反復可能な node id 群。node_id が葉なら変化なし。
    呼び出し側で is_leaf を確認すること (ここでは children を tree 無しに判定不可)。
    この関数はツリー非依存のため、子 id を別途渡せないので PartitionTree 版を使う。
    """
    raise NotImplementedError("split_visible_node_tree を使用してください")


def split_visible_node_tree(tree: PartitionTree, visible_nodes, node_id: int) -> list[int]:
    """表示集合の node_id を子へ分割した新しい昇順リストを返す。"""
    vis = set(int(v) for v in visible_nodes)
    nid = int(node_id)
    if nid not in vis or tree.is_leaf(nid):
        return sorted(vis)
    l, r = tree.children(nid)
    vis.discard(nid)
    vis.add(l)
    vis.add(r)
    return sorted(vis)


def collapse_siblings(visible_nodes, node_a: int, node_b: int,
                      tree: PartitionTree | None = None) -> list[int]:
    """
    兄弟ノード node_a, node_b を親へ統合表示する (親へ戻す)。

    tree が与えられた場合は親子関係を検証する。兄弟でなければ変化なし。
    """
    vis = set(int(v) for v in visible_nodes)
    a, b = int(node_a), int(node_b)
    if tree is not None:
        pa, pb = tree.parent_of(a), tree.parent_of(b)
        if pa == 0 or pa != pb:
            return sorted(vis)
        parent = pa
    else:
        return sorted(vis)
    if a in vis and b in vis:
        vis.discard(a)
        vis.discard(b)
        vis.add(parent)
    return sorted(vis)


def collapse_to_parent(tree: PartitionTree, visible_nodes, node_id: int) -> list[int]:
    """node_id とその兄弟を親へ戻す (Backspace 操作)。"""
    vis = set(int(v) for v in visible_nodes)
    nid = int(node_id)
    parent = tree.parent_of(nid)
    if parent == 0:
        return sorted(vis)
    l, r = tree.children(parent)
    sib = r if l == nid else l
    vis.discard(nid)
    vis.discard(sib)
    vis.add(parent)
    return sorted(vis)


def leaf_to_visible_node(leaf_id: int, visible_nodes, parent_array) -> int:
    """
    leaf_id から root 方向へ parent をたどり、visible に含まれる祖先ノードを返す。

    parent_array は index = node_id - 1 の親配列。見つからなければ最後に辿った
    (parent=0 の) ノードを返す。
    """
    visible = set(int(v) for v in visible_nodes)
    parent = np.asarray(parent_array, dtype=np.int64)
    cur = int(leaf_id)
    while cur != 0:
        if cur in visible:
            return cur
        nxt = int(parent[cur - 1])
        if nxt == 0:
            return cur
        cur = nxt
    return int(leaf_id)
