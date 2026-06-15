"""V0.10: REMOVE_ONLY 判断 Undo 履歴のテスト。"""

from ai.amg_review_history import (
    DEFAULT_UNDO_LIMIT,
    ReviewAction,
    ReviewHistory,
    apply_undo,
)


def test_record_and_undo_single():
    h = ReviewHistory()
    h.record(ReviewAction(5, "unreviewed", "remove"))
    assert h.can_undo()
    step = h.undo()
    assert len(step) == 1 and step[0].segment_id == 5
    assert not h.can_undo()


def test_apply_undo_restores_before():
    decisions = {"5": "remove"}
    step = [ReviewAction(5, "unreviewed", "remove")]
    restored = apply_undo(decisions, step)
    assert "5" not in restored        # before=unreviewed -> キー削除


def test_undo_release_restores_remove():
    decisions = {}     # 解除後 (remove が消えた状態)
    step = [ReviewAction(7, "remove", "unreviewed")]
    restored = apply_undo(decisions, step)
    assert restored["7"] == "remove"


def test_batch_is_single_step():
    h = ReviewHistory()
    h.record([ReviewAction(1, "unreviewed", "remove"),
              ReviewAction(2, "unreviewed", "remove"),
              ReviewAction(3, "unreviewed", "remove")])
    assert len(h) == 1
    step = h.undo()
    assert len(step) == 3
    decisions = {"1": "remove", "2": "remove", "3": "remove"}
    restored = apply_undo(decisions, step)
    assert restored == {}


def test_clear_all_remove_as_one_step():
    h = ReviewHistory()
    # 全 REMOVE 解除を 1 ステップで記録
    h.record([ReviewAction(1, "remove", "unreviewed"),
              ReviewAction(4, "remove", "unreviewed")])
    assert len(h) == 1
    restored = apply_undo({}, h.undo())
    assert restored == {"1": "remove", "4": "remove"}


def test_no_op_actions_skipped():
    h = ReviewHistory()
    h.record([ReviewAction(1, "remove", "remove")])   # 変化なし
    assert len(h) == 0
    assert not h.can_undo()


def test_limit_drops_oldest():
    h = ReviewHistory(limit=DEFAULT_UNDO_LIMIT)
    assert h.limit == 100
    for i in range(120):
        h.record(ReviewAction(i, "unreviewed", "remove"))
    assert len(h) == 100
    # 最後に記録したものが残っている
    step = h.undo()
    assert step[0].segment_id == 119


def test_undo_empty_returns_empty():
    h = ReviewHistory()
    assert h.undo() == []
