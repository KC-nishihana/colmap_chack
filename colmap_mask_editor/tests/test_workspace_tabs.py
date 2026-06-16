"""V0.11: 右パネルのワークスペース2タブ (レビュー / プロジェクト処理) と統合ツールバー配置のテスト。

既存の 4 タブ (_right_tab_widget) は「レビュー」配下へ入れ子化される (非破壊)。
"""

import pytest
from PySide6.QtWidgets import QTabWidget

from core.selection_tools import ApplyOperation, SelectionTool
from ui.main_window import MainWindow
from ui.unified_tool_bar import UnifiedToolBar


def test_workspace_tabs_exist(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    assert hasattr(win, "_workspace_tabs")
    assert isinstance(win._workspace_tabs, QTabWidget)
    assert win._workspace_tabs.count() == 2


def test_workspace_tab_labels(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    labels = [win._workspace_tabs.tabText(i) for i in range(win._workspace_tabs.count())]
    assert labels == ["レビュー", "プロジェクト処理"]


def test_existing_4_tabs_nested_and_intact(qtbot):
    # 既存 4 タブは維持 (レビュー配下に入れ子)。回帰しないこと。
    win = MainWindow()
    qtbot.addWidget(win)
    assert win._right_tab_widget.count() == 4
    labels = [win._right_tab_widget.tabText(i) for i in range(4)]
    assert labels == ["編集", "GrabCut", "AIセグメント", "保存・確認"]


def test_unified_tool_bar_in_top_toolbar(qtbot):
    # 統合ツールバーは横幅確保のためウィンドウ上部 (全幅) のツールバーへ置く。
    from PySide6.QtWidgets import QToolBar
    win = MainWindow()
    qtbot.addWidget(win)
    assert hasattr(win, "_unified_tool_bar")
    assert isinstance(win._unified_tool_bar, UnifiedToolBar)
    tb = win.findChild(QToolBar, "unified_toolbar")
    assert tb is not None                                  # トップツールバーに存在
    assert win._unified_tool_bar.parent() is not None
    # 既定: AIクリック / 除外する
    assert win._selection_tool is SelectionTool.AI_CLICK
    assert win._apply_operation is ApplyOperation.REMOVE


def test_toolbar_signals_update_main_window_state(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    win._unified_tool_bar.set_selection_tool(SelectionTool.BRUSH)
    assert win._selection_tool is SelectionTool.BRUSH
    win._unified_tool_bar.set_apply_operation(ApplyOperation.ADD)
    assert win._apply_operation is ApplyOperation.ADD


def test_legacy_amg_review_button_in_project_tab(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    assert hasattr(win, "_btn_legacy_amg_review")
    assert "AMG" in win._btn_legacy_amg_review.text()


def test_workspace_restored_from_settings(qtbot, tmp_path, monkeypatch):
    from core.app_settings import AppSettings
    ini_path = str(tmp_path / "settings.ini")
    s = AppSettings(filepath=ini_path)
    s.save({"ui/main_workspace": "project"})
    monkeypatch.setattr("ui.main_window.AppSettings",
                        lambda: AppSettings(filepath=ini_path))
    win = MainWindow()
    qtbot.addWidget(win)
    assert win._workspace_tabs.currentIndex() == 1   # プロジェクト処理
