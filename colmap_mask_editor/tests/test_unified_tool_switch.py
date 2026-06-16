"""V0.11 B-2: 統合ツールバー ↔ 中央キャンバス / 編集タブラジオの双方向同期テスト。"""

import pytest

from core.selection_tools import ApplyOperation, SelectionTool
from ui.image_canvas import EditMode
from ui.main_window import MainWindow


def test_toolbar_brush_sets_canvas_and_radio(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    win._unified_tool_bar.set_selection_tool(SelectionTool.BRUSH)
    assert win._canvas.get_edit_mode() is EditMode.BRUSH
    assert win._mode_btns[EditMode.BRUSH].isChecked()


def test_toolbar_rectangle_with_remove_maps_to_rect_del(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    win._unified_tool_bar.set_apply_operation(ApplyOperation.REMOVE)
    win._unified_tool_bar.set_selection_tool(SelectionTool.RECTANGLE)
    assert win._canvas.get_edit_mode() is EditMode.RECT_DEL


def test_toolbar_rectangle_with_add_maps_to_rect_add(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    win._unified_tool_bar.set_apply_operation(ApplyOperation.ADD)
    win._unified_tool_bar.set_selection_tool(SelectionTool.RECTANGLE)
    assert win._canvas.get_edit_mode() is EditMode.RECT_ADD


def test_changing_operation_reswitches_rect_mode(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    win._unified_tool_bar.set_selection_tool(SelectionTool.RECTANGLE)
    win._unified_tool_bar.set_apply_operation(ApplyOperation.ADD)
    assert win._canvas.get_edit_mode() is EditMode.RECT_ADD
    win._unified_tool_bar.set_apply_operation(ApplyOperation.REMOVE)
    assert win._canvas.get_edit_mode() is EditMode.RECT_DEL


def test_radio_change_reflects_into_toolbar(qtbot):
    # 編集タブのラジオ -> ツールバーへ反映 (双方向)
    win = MainWindow()
    qtbot.addWidget(win)
    win._set_mode(EditMode.POLY_ADD)
    assert win._unified_tool_bar.selection_tool() is SelectionTool.POLYGON
    assert win._unified_tool_bar.apply_operation() is ApplyOperation.ADD
    assert win._selection_tool is SelectionTool.POLYGON


def test_pan_syncs_both_ways(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    win._unified_tool_bar.set_selection_tool(SelectionTool.PAN)
    assert win._canvas.get_edit_mode() is EditMode.PAN
    win._set_mode(EditMode.BRUSH)
    assert win._unified_tool_bar.selection_tool() is SelectionTool.BRUSH


def test_grabcut_mode_leaves_toolbar_unchanged(qtbot):
    # GrabCut は統合ツールバーに対応が無い -> ツールバーは変わらない (クラッシュしない)
    win = MainWindow()
    qtbot.addWidget(win)
    win._unified_tool_bar.set_selection_tool(SelectionTool.BRUSH)
    before = win._unified_tool_bar.selection_tool()
    win._set_mode(EditMode.GRABCUT_ADD)
    assert win._canvas.get_edit_mode() is EditMode.GRABCUT_ADD
    assert win._unified_tool_bar.selection_tool() is before   # 変化なし


def test_ai_automatic_does_not_crash_or_change_mode(qtbot):
    # AI_AUTOMATIC は対応 EditMode 無し -> mode は変えず状態だけ保持 (B-3 で結線)
    win = MainWindow()
    qtbot.addWidget(win)
    win._set_mode(EditMode.BRUSH)
    win._unified_tool_bar.set_selection_tool(SelectionTool.AI_AUTOMATIC)
    assert win._selection_tool is SelectionTool.AI_AUTOMATIC
    assert win._canvas.get_edit_mode() is EditMode.BRUSH   # mode は据え置き


def test_no_infinite_loop_on_repeated_toggles(qtbot):
    # 双方向同期で再帰ループしないこと (emit=False 反映)。例外なく完了すれば良い。
    win = MainWindow()
    qtbot.addWidget(win)
    for _ in range(5):
        win._unified_tool_bar.set_selection_tool(SelectionTool.RECTANGLE)
        win._unified_tool_bar.set_selection_tool(SelectionTool.BRUSH)
        win._set_mode(EditMode.POLY_DEL)
    assert win._canvas.get_edit_mode() is EditMode.POLY_DEL
