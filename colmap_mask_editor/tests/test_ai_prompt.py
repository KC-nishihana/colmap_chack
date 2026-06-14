"""
AIプロンプト (正/負クリック・矩形) と Undo/Redo のテスト。
"""

from ai.ai_prompt import (
    AiBoxPrompt,
    AiPromptSession,
    LABEL_NEGATIVE,
    LABEL_POSITIVE,
)


def test_add_positive_and_negative_points():
    s = AiPromptSession()
    s.add_point(100, 200, positive=True)
    s.add_point(50, 60, positive=False)
    pts = s.points
    assert len(pts) == 2
    assert pts[0].label == LABEL_POSITIVE
    assert pts[1].label == LABEL_NEGATIVE


def test_set_box_normalizes():
    s = AiPromptSession()
    s.set_box(900, 1000, 50, 80)  # 右下→左上の順
    b = s.box
    assert b.x1 == 50 and b.y1 == 80
    assert b.x2 == 900 and b.y2 == 1000


def test_prompt_undo_redo_independent():
    s = AiPromptSession()
    s.add_point(10, 10, positive=True)
    s.add_point(20, 20, positive=True)
    assert len(s.points) == 2

    assert s.undo() is True
    assert len(s.points) == 1
    assert s.undo() is True
    assert len(s.points) == 0
    assert s.undo() is False  # これ以上戻せない

    assert s.redo() is True
    assert len(s.points) == 1
    assert s.redo() is True
    assert len(s.points) == 2


def test_remove_last_point():
    s = AiPromptSession()
    s.add_point(1, 1, positive=True)
    s.add_point(2, 2, positive=True)
    assert s.remove_last_point() is True
    assert len(s.points) == 1
    s.remove_last_point()
    assert s.remove_last_point() is False


def test_clear_is_undoable_but_reset_is_not():
    s = AiPromptSession()
    s.add_point(1, 1, positive=True)
    s.clear()
    assert s.is_empty()
    assert s.undo() is True  # clear を戻せる
    assert len(s.points) == 1

    s.reset()
    assert s.is_empty()
    assert s.undo() is False  # reset は履歴ごと消える


def test_to_predict_fields():
    s = AiPromptSession()
    s.add_point(100, 200, positive=True)
    s.add_point(400, 300, positive=False)
    s.set_box(50, 80, 900, 1000)
    fields = s.to_predict_fields()
    assert fields["points"] == [
        {"x": 100.0, "y": 200.0, "label": 1},
        {"x": 400.0, "y": 300.0, "label": 0},
    ]
    assert fields["box"] == [50.0, 80.0, 900.0, 1000.0]


def test_to_predict_fields_without_box():
    s = AiPromptSession()
    s.add_point(1, 1, positive=True)
    fields = s.to_predict_fields()
    assert "box" not in fields


def test_box_width_height():
    b = AiBoxPrompt(10, 20, 110, 220)
    assert b.width == 100
    assert b.height == 200


def test_redo_cleared_after_new_edit():
    s = AiPromptSession()
    s.add_point(1, 1, positive=True)
    s.add_point(2, 2, positive=True)
    s.undo()
    assert s.can_redo()
    s.add_point(3, 3, positive=True)  # 新しい編集で redo は消える
    assert not s.can_redo()
