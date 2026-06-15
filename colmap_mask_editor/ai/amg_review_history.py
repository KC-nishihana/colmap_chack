"""
V0.10: REMOVE_ONLY レビューの判断 Undo 履歴 (numpy / Qt 非依存)。

候補判断 (REMOVE / 解除) の Undo は通常マスクの Undo とは分離する。1 操作 = 1 ステップ
で、一括 REMOVE / 一括解除 / 全 REMOVE 解除も 1 ステップとして 1 回の Undo で戻せる。

最大履歴数は 100 (古いステップから捨てる)。redo は持たない (要件は Undo のみ)。
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_UNDO_LIMIT = 100

__all__ = ["ReviewAction", "ReviewHistory", "DEFAULT_UNDO_LIMIT", "apply_undo"]


@dataclass(frozen=True)
class ReviewAction:
    """1 候補の判断変更 (before -> after)。"""
    segment_id: int
    before: str
    after: str


class ReviewHistory:
    """判断変更ステップのスタック。各ステップは 1 個以上の ReviewAction。"""

    def __init__(self, limit: int = DEFAULT_UNDO_LIMIT) -> None:
        self._limit = max(1, int(limit))
        self._steps: list[list[ReviewAction]] = []

    def record(self, actions) -> None:
        """1 ステップを記録する。actions は ReviewAction か ReviewAction の列。

        実際に変化のない action (before == after) は除外する。空ステップは積まない。
        """
        if isinstance(actions, ReviewAction):
            actions = [actions]
        step = [a for a in actions if a.before != a.after]
        if not step:
            return
        self._steps.append(list(step))
        while len(self._steps) > self._limit:
            self._steps.pop(0)

    def can_undo(self) -> bool:
        return bool(self._steps)

    def undo(self) -> list[ReviewAction]:
        """直前のステップを取り出して返す (戻す処理は呼び出し側)。無ければ空リスト。"""
        if not self._steps:
            return []
        return self._steps.pop()

    def clear(self) -> None:
        self._steps.clear()

    @property
    def limit(self) -> int:
        return self._limit

    def __len__(self) -> int:
        return len(self._steps)


def apply_undo(decisions: dict[str, str], step: list[ReviewAction]) -> dict[str, str]:
    """
    Undo ステップを decisions へ適用した新しい dict を返す (各候補を before へ戻す)。

    before が 'unreviewed' の候補はキーを削除して REMOVE_ONLY の最小保存を保つ。
    """
    out = dict(decisions)
    for action in step:
        key = str(int(action.segment_id))
        if action.before == "unreviewed":
            out.pop(key, None)
        else:
            out[key] = action.before
    return out
