"""
V0.9: partition レビューの状態モデル (Qt 非依存・numpy のみ)。

階層カット (粒度)、局所 split / collapse、ノード判断と継承、画素率集計、次の未確認
移動を担う。表示カットと判断は独立 (粒度変更で判断を失わない)。レビュー widget は
この model を介して状態を操作し、partition_review.json へは別途保存する。
"""

from __future__ import annotations

import numpy as np

from ai import partition_review_state as prs
from ai.partition_review_state import RegionDecision
from ai.partition_tree import (
    PartitionTree,
    collapse_to_parent,
    cut_tree_to_count,
    split_visible_node_tree,
)

__all__ = ["PartitionReviewModel"]


class PartitionReviewModel:
    def __init__(self, partition_data: dict, review: dict | None = None):
        self.data = partition_data
        self.tree = PartitionTree.from_npz(partition_data)
        self.node_area = np.asarray(partition_data["node_area"])
        self.parent = np.asarray(partition_data["node_parent"])
        review = review or {}
        self.decisions: dict[str, str] = prs.normalize_node_decisions(
            review.get("node_decisions", {}), self.tree.node_count)
        self.target_visible_count = int(
            review.get("target_visible_count", self.tree.leaf_count))
        self.visible: list[int] = cut_tree_to_count(self.tree, self.target_visible_count)
        self.last_selected_node = review.get("last_selected_node")

    # ---- 粒度 ---- #
    def set_target_visible_count(self, count: int) -> None:
        """粒度変更。判断は保持 (decisions に触れない)。"""
        self.target_visible_count = max(1, int(count))
        self.visible = cut_tree_to_count(self.tree, self.target_visible_count)

    # ---- 局所操作 ---- #
    def split(self, node_id: int) -> bool:
        """選択ノードを子へ 1 段階細分化。葉なら何もしない。"""
        if self.tree.is_leaf(node_id):
            return False
        self.visible = split_visible_node_tree(self.tree, self.visible, node_id)
        return True

    def collapse(self, node_id: int) -> int | None:
        """選択ノードと兄弟を親へ戻す。戻した親 id を返す。"""
        parent = self.tree.parent_of(node_id)
        if parent == 0:
            return None
        self.visible = collapse_to_parent(self.tree, self.visible, node_id)
        return parent

    # ---- 判断 ---- #
    def effective(self, node_id: int) -> str:
        return prs.effective_decision(self.parent, node_id, self.decisions)

    def descendants_with_decisions(self, node_id: int) -> list[int]:
        return prs.descendant_explicit_nodes(
            self.data["node_left"], self.data["node_right"], node_id, self.decisions)

    def set_decision(self, node_id: int, decision: RegionDecision,
                     clear_descendants: bool = False) -> None:
        if clear_descendants:
            self.decisions = prs.set_parent_decision_clearing_descendants(
                self.data["node_left"], self.data["node_right"],
                self.decisions, node_id, decision)
        else:
            self.decisions = prs.set_node_decision(self.decisions, node_id, decision)

    def set_descendants_decision(self, node_id: int, decision: RegionDecision) -> None:
        """選択領域の子孫の葉をすべて keep/remove にする (一括判断)。"""
        # 親に設定し子孫の明示判断を消すのと等価 (継承で全葉が同判断になる)
        self.set_decision(node_id, decision, clear_descendants=True)

    # ---- 画素率・統計 ---- #
    def pixel_rates(self) -> dict:
        return prs.pixel_rates(self.parent, self.tree.leaf_count,
                               self.node_area, self.decisions)

    def stats(self) -> dict:
        reviewed = sum(1 for n in self.visible if self.effective(n) != "unreviewed")
        rates = self.pixel_rates()
        return {
            "visible_count": len(self.visible),
            "leaf_count": self.tree.leaf_count,
            "reviewed_visible": reviewed,
            "keep_ratio": rates["keep_ratio"],
            "remove_ratio": rates["remove_ratio"],
            "unreviewed_ratio": rates["unreviewed_ratio"],
            "assigned_ratio": rates["assigned_ratio"],
        }

    def is_complete(self) -> bool:
        return prs.is_complete(self.parent, self.tree.leaf_count, self.decisions)

    # ---- 次の未確認 ---- #
    def next_unreviewed(self, current: int | None, forward: bool = True) -> int | None:
        """表示ノードのうち実効判断が未確認のものを巡回する。"""
        unrev = [n for n in self.visible if self.effective(n) == "unreviewed"]
        if not unrev:
            return None
        if current is None or current not in unrev:
            return unrev[0] if forward else unrev[-1]
        pos = unrev.index(current)
        pos = (pos + 1) % len(unrev) if forward else (pos - 1) % len(unrev)
        return unrev[pos]

    # ---- 一括選択ヘルパー ---- #
    def visible_touching_border(self) -> list[int]:
        """画像端に接する表示ノード。"""
        bbox = np.asarray(self.data["node_bbox"])
        h, w = (int(self.data["image_shape"][0]), int(self.data["image_shape"][1]))
        out = []
        for n in self.visible:
            x, y, bw, bh = bbox[n - 1].tolist()
            if x <= 0 or y <= 0 or x + bw >= w or y + bh >= h:
                out.append(n)
        return out

    def visible_smaller_than(self, max_area: int) -> list[int]:
        return [n for n in self.visible if int(self.node_area[n - 1]) <= int(max_area)]

    def visible_unreviewed(self) -> list[int]:
        return [n for n in self.visible if self.effective(n) == "unreviewed"]

    def visible_siblings_of(self, node_id: int) -> list[int]:
        parent = self.tree.parent_of(node_id)
        if parent == 0:
            return []
        l, r = self.tree.children(parent)
        return [n for n in (l, r) if n in set(self.visible)]

    # ---- 永続化用 ---- #
    def review_dict(self, partition_sha: str, completed: bool | None = None) -> dict:
        from ai import partition_manifest as pman
        return pman.build_partition_review(
            partition_npz_sha256=partition_sha,
            target_visible_count=self.target_visible_count,
            node_decisions=self.decisions,
            completed=bool(completed) if completed is not None else self.is_complete(),
            last_selected_node=self.last_selected_node,
        )
