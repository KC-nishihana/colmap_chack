"""V0.11: 統合ツールバー (UnifiedToolBar) のテスト。選択ツールと適用操作の分離。"""

import pytest

from core.selection_tools import ApplyOperation, SelectionTool
from ui.unified_tool_bar import UnifiedToolBar


class _FakeSettings:
    """AppSettings.get(key, default) 互換の最小スタブ。"""
    def __init__(self, values=None):
        self._v = dict(values or {})

    def get(self, key, default=None):
        return self._v.get(key, default)


def test_defaults_ai_click_and_remove(qtbot):
    tb = UnifiedToolBar()
    qtbot.addWidget(tb)
    assert tb.selection_tool() is SelectionTool.AI_CLICK
    assert tb.apply_operation() is ApplyOperation.REMOVE


def test_selecting_tool_emits_and_updates(qtbot):
    tb = UnifiedToolBar()
    qtbot.addWidget(tb)
    with qtbot.waitSignal(tb.selection_tool_changed, timeout=1000) as sig:
        tb._tool_btns[SelectionTool.BRUSH].click()
    assert sig.args[0] is SelectionTool.BRUSH
    assert tb.selection_tool() is SelectionTool.BRUSH
    # 排他: ブラシだけ checked
    assert tb._tool_btns[SelectionTool.BRUSH].isChecked()
    assert not tb._tool_btns[SelectionTool.AI_CLICK].isChecked()


def test_selecting_operation_emits_and_updates(qtbot):
    tb = UnifiedToolBar()
    qtbot.addWidget(tb)
    with qtbot.waitSignal(tb.apply_operation_changed, timeout=1000) as sig:
        tb._op_btns[ApplyOperation.ADD].click()
    assert sig.args[0] is ApplyOperation.ADD
    assert tb.apply_operation() is ApplyOperation.ADD


def test_tools_are_mutually_exclusive(qtbot):
    tb = UnifiedToolBar()
    qtbot.addWidget(tb)
    tb._tool_btns[SelectionTool.RECTANGLE].click()
    tb._tool_btns[SelectionTool.POLYGON].click()
    checked = [t for t, b in tb._tool_btns.items() if b.isChecked()]
    assert checked == [SelectionTool.POLYGON]


def test_set_selection_tool_emit_false_is_silent(qtbot):
    tb = UnifiedToolBar()
    qtbot.addWidget(tb)
    received = []
    tb.selection_tool_changed.connect(received.append)
    tb.set_selection_tool(SelectionTool.POLYGON, emit=False)
    assert tb.selection_tool() is SelectionTool.POLYGON
    assert received == []   # シグナルは出ない


def test_set_apply_operation_emit_true_signals(qtbot):
    tb = UnifiedToolBar()
    qtbot.addWidget(tb)
    with qtbot.waitSignal(tb.apply_operation_changed, timeout=1000) as sig:
        tb.set_apply_operation(ApplyOperation.REPLACE, emit=True)
    assert sig.args[0] is ApplyOperation.REPLACE


def test_settings_override_defaults(qtbot):
    s = _FakeSettings({
        "ui/default_selection_tool": "brush",
        "ui/default_apply_operation": "add",
    })
    tb = UnifiedToolBar(settings=s)
    qtbot.addWidget(tb)
    assert tb.selection_tool() is SelectionTool.BRUSH
    assert tb.apply_operation() is ApplyOperation.ADD


def test_hide_ai_buttons_via_settings(qtbot):
    s = _FakeSettings({
        "ui/show_ai_click_button": False,
        "ui/show_ai_automatic_button": False,
        # 既定ツールが非表示にならないよう brush を既定にする
        "ui/default_selection_tool": "brush",
    })
    tb = UnifiedToolBar(settings=s)
    qtbot.addWidget(tb)
    tb.show()
    assert not tb._tool_btns[SelectionTool.AI_CLICK].isVisible()
    assert not tb._tool_btns[SelectionTool.AI_AUTOMATIC].isVisible()
    assert tb._tool_btns[SelectionTool.BRUSH].isVisible()


def test_invalid_settings_fall_back_to_defaults(qtbot):
    s = _FakeSettings({"ui/default_selection_tool": "bogus"})
    tb = UnifiedToolBar(settings=s)
    qtbot.addWidget(tb)
    assert tb.selection_tool() is SelectionTool.AI_CLICK
