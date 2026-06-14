"""
右パネルタブ構成のテスト (v0.5.1)

QTabWidgetの存在・タブラベル・常時表示ボタン・自動切替を確認する。
"""

import pytest
from PySide6.QtWidgets import QTabWidget

from ui.image_canvas import EditMode
from ui.main_window import MainWindow


# ------------------------------------------------------------------ #
# QTabWidget の存在確認
# ------------------------------------------------------------------ #

def test_right_tab_widget_exists(qtbot):
    """右パネルに QTabWidget が存在する"""
    win = MainWindow()
    qtbot.addWidget(win)
    assert hasattr(win, "_right_tab_widget")
    assert isinstance(win._right_tab_widget, QTabWidget)


def test_tab_count_is_4(qtbot):
    """タブが4つある (v0.6: 編集/GrabCut/AIセグメント/保存・確認)"""
    win = MainWindow()
    qtbot.addWidget(win)
    assert win._right_tab_widget.count() == 4


def test_ai_tab_exists(qtbot):
    """「AIセグメント」タブが存在する"""
    win = MainWindow()
    qtbot.addWidget(win)
    labels = [win._right_tab_widget.tabText(i) for i in range(win._right_tab_widget.count())]
    assert "AIセグメント" in labels


def test_edit_tab_exists(qtbot):
    """「編集」タブが存在する"""
    win = MainWindow()
    qtbot.addWidget(win)
    labels = [win._right_tab_widget.tabText(i) for i in range(win._right_tab_widget.count())]
    assert "編集" in labels


def test_grabcut_tab_exists(qtbot):
    """「GrabCut」タブが存在する"""
    win = MainWindow()
    qtbot.addWidget(win)
    labels = [win._right_tab_widget.tabText(i) for i in range(win._right_tab_widget.count())]
    assert "GrabCut" in labels


def test_save_tab_exists(qtbot):
    """「保存・確認」タブが存在する"""
    win = MainWindow()
    qtbot.addWidget(win)
    labels = [win._right_tab_widget.tabText(i) for i in range(win._right_tab_widget.count())]
    assert "保存・確認" in labels


# ------------------------------------------------------------------ #
# 常時表示ボタンの存在確認
# ------------------------------------------------------------------ #

def test_prev_button_always_visible(qtbot):
    """「前の画像」ボタンが存在する"""
    win = MainWindow()
    qtbot.addWidget(win)
    assert hasattr(win, "_btn_prev")
    assert win._btn_prev is not None


def test_next_button_always_visible(qtbot):
    """「次の画像」ボタンが存在する"""
    win = MainWindow()
    qtbot.addWidget(win)
    assert hasattr(win, "_btn_next")
    assert win._btn_next is not None


def test_save_button_always_visible(qtbot):
    """「保存」ボタンが存在する"""
    win = MainWindow()
    qtbot.addWidget(win)
    assert hasattr(win, "_btn_save")
    assert win._btn_save is not None


def test_undo_redo_buttons_always_visible(qtbot):
    """Undo/Redo ボタンが存在する"""
    win = MainWindow()
    qtbot.addWidget(win)
    assert hasattr(win, "_btn_undo")
    assert hasattr(win, "_btn_redo")


# ------------------------------------------------------------------ #
# タブ自動切替
# ------------------------------------------------------------------ #

def test_grabcut_mode_switches_to_grabcut_tab(qtbot):
    """GrabCut系モード選択時にGrabCutタブへ切り替わる"""
    win = MainWindow()
    qtbot.addWidget(win)
    win._right_tab_widget.setCurrentIndex(0)  # 編集タブにしておく

    win._set_mode(EditMode.GRABCUT_ADD)

    assert win._right_tab_widget.currentIndex() == 1  # GrabCutタブ


def test_brush_mode_switches_to_edit_tab(qtbot):
    """ブラシモード選択時に編集タブへ切り替わる"""
    win = MainWindow()
    qtbot.addWidget(win)
    win._right_tab_widget.setCurrentIndex(1)  # GrabCutタブにしておく

    win._set_mode(EditMode.BRUSH)

    assert win._right_tab_widget.currentIndex() == 0  # 編集タブ


def test_shift_g_mode_switches_to_grabcut_tab(qtbot):
    """Shift+G (GRABCUT_DEL) モードでGrabCutタブへ切り替わる"""
    win = MainWindow()
    qtbot.addWidget(win)
    win._right_tab_widget.setCurrentIndex(0)

    win._set_mode(EditMode.GRABCUT_DEL)

    assert win._right_tab_widget.currentIndex() == 1


def test_poly_mode_switches_to_edit_tab(qtbot):
    """ポリゴンモード選択時に編集タブへ切り替わる"""
    win = MainWindow()
    qtbot.addWidget(win)
    win._right_tab_widget.setCurrentIndex(1)

    win._set_mode(EditMode.POLY_ADD)

    assert win._right_tab_widget.currentIndex() == 0


# ------------------------------------------------------------------ #
# 設定からのタブ番号復元 (起動時)
# ------------------------------------------------------------------ #

def test_tab_index_restored_from_settings(qtbot, tmp_path, monkeypatch):
    """起動時に保存されたタブ番号が復元される"""
    from core.app_settings import AppSettings

    ini_path = str(tmp_path / "settings.ini")
    s = AppSettings(filepath=ini_path)
    s.save({"window/right_tab_index": 2})  # 保存・確認タブ

    # MainWindow が使う AppSettings を差し替え
    monkeypatch.setattr(
        "ui.main_window.AppSettings",
        lambda: AppSettings(filepath=ini_path),
    )

    win = MainWindow()
    qtbot.addWidget(win)

    assert win._right_tab_widget.currentIndex() == 2


# ------------------------------------------------------------------ #
# 既存属性の維持確認
# ------------------------------------------------------------------ #

def test_grabcut_iter_spin_exists_in_tab(qtbot):
    """_grabcut_iter_spin 属性が維持されている"""
    win = MainWindow()
    qtbot.addWidget(win)
    assert hasattr(win, "_grabcut_iter_spin")
    assert win._grabcut_iter_spin.value() == 5


def test_gc_correction_group_exists_in_tab(qtbot):
    """_gc_correction_group 属性が維持されている"""
    win = MainWindow()
    qtbot.addWidget(win)
    assert hasattr(win, "_gc_correction_group")


def test_canvas_attribute_exists(qtbot):
    """_canvas 属性が維持されている"""
    win = MainWindow()
    qtbot.addWidget(win)
    assert hasattr(win, "_canvas")
    assert win._canvas is not None


def test_main_splitter_exists(qtbot):
    """_main_splitter 属性が存在する"""
    win = MainWindow()
    qtbot.addWidget(win)
    assert hasattr(win, "_main_splitter")
