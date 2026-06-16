"""V0.11: SelectionTool / ApplyOperation 分離と EditMode 後方互換アダプタのテスト。"""

import pytest

from core.selection_tools import (
    DEFAULT_APPLY_OPERATION,
    DEFAULT_SELECTION_TOOL,
    ApplyOperation,
    SelectionTool,
    from_edit_mode,
    to_edit_mode,
)


def test_enum_values():
    assert SelectionTool.AI_CLICK.value == "ai_click"
    assert SelectionTool.AI_AUTOMATIC.value == "ai_automatic"
    assert {t.value for t in SelectionTool} == {
        "ai_click", "ai_automatic", "brush", "polygon", "rectangle", "pan"}
    assert {o.value for o in ApplyOperation} == {"add", "remove", "replace"}


def test_defaults_match_spec():
    assert DEFAULT_SELECTION_TOOL is SelectionTool.AI_CLICK
    assert DEFAULT_APPLY_OPERATION is ApplyOperation.REMOVE


def test_from_str_roundtrip():
    for t in SelectionTool:
        assert SelectionTool.from_str(t.value) is t
    for o in ApplyOperation:
        assert ApplyOperation.from_str(o.value) is o
    with pytest.raises(ValueError):
        SelectionTool.from_str("nope")
    with pytest.raises(ValueError):
        ApplyOperation.from_str("nope")


# ---- EditMode アダプタ (PySide6 が必要) ----

def test_to_edit_mode_rectangle_polygon():
    from ui.image_canvas import EditMode
    assert to_edit_mode(SelectionTool.RECTANGLE, ApplyOperation.ADD) is EditMode.RECT_ADD
    assert to_edit_mode(SelectionTool.RECTANGLE, ApplyOperation.REMOVE) is EditMode.RECT_DEL
    assert to_edit_mode(SelectionTool.POLYGON, ApplyOperation.ADD) is EditMode.POLY_ADD
    assert to_edit_mode(SelectionTool.POLYGON, ApplyOperation.REMOVE) is EditMode.POLY_DEL
    assert to_edit_mode(SelectionTool.BRUSH, ApplyOperation.ADD) is EditMode.BRUSH
    assert to_edit_mode(SelectionTool.PAN, ApplyOperation.REMOVE) is EditMode.PAN
    assert to_edit_mode(SelectionTool.AI_CLICK, ApplyOperation.REMOVE) is EditMode.AI_PROMPT


def test_to_edit_mode_unmappable_raises():
    # AI_AUTOMATIC と REPLACE は旧 EditMode に対応値が無い
    with pytest.raises(ValueError):
        to_edit_mode(SelectionTool.AI_AUTOMATIC, ApplyOperation.REMOVE)
    with pytest.raises(ValueError):
        to_edit_mode(SelectionTool.RECTANGLE, ApplyOperation.REPLACE)


def test_from_edit_mode():
    from ui.image_canvas import EditMode
    assert from_edit_mode(EditMode.RECT_ADD) == (SelectionTool.RECTANGLE, ApplyOperation.ADD)
    assert from_edit_mode(EditMode.POLY_DEL) == (SelectionTool.POLYGON, ApplyOperation.REMOVE)
    assert from_edit_mode(EditMode.AI_PROMPT)[0] is SelectionTool.AI_CLICK
    # GrabCut 系は新ツールに対応が無い -> None
    assert from_edit_mode(EditMode.GRABCUT_ADD) is None
