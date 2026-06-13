"""
MainWindow GUIスモークテスト (v0.4A.1)

実画像ファイルやCOLMAPプロジェクトに依存せず、
ウィンドウの生成・初期値・クローズのみを確認する。
"""

import pytest
from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow


# ------------------------------------------------------------------ #
# スモークテスト
# ------------------------------------------------------------------ #

def test_main_window_creates_without_error(qtbot):
    """MainWindowをプロジェクトなしで生成できる"""
    win = MainWindow()
    qtbot.addWidget(win)
    assert win is not None


def test_grabcut_iter_initial_value(qtbot):
    """GrabCut反復回数の初期値が5"""
    win = MainWindow()
    qtbot.addWidget(win)
    assert win._grabcut_iter_spin.value() == 5


def test_grabcut_max_size_initial_value(qtbot):
    """GrabCut最大処理サイズの初期値が2048"""
    win = MainWindow()
    qtbot.addWidget(win)
    assert win._grabcut_max_size_spin.value() == 2048


def test_grabcut_use_downscale_initial_true(qtbot):
    """大画像縮小処理の初期値がTrue"""
    win = MainWindow()
    qtbot.addWidget(win)
    assert win._grabcut_use_downscale_cb.isChecked() is True


def test_grabcut_processing_flag_initial_false(qtbot):
    """GrabCut処理中フラグの初期値がFalse"""
    win = MainWindow()
    qtbot.addWidget(win)
    assert win._canvas.grabcut_processing is False


def test_no_grabcut_worker_on_startup(qtbot):
    """起動時にGrabCutワーカーが存在しない"""
    win = MainWindow()
    qtbot.addWidget(win)
    assert win._grabcut_worker is None
    assert win._grabcut_thread is None


def test_window_closes_without_error(qtbot):
    """ウィンドウを例外なく閉じられる"""
    win = MainWindow()
    qtbot.addWidget(win)
    win.close()


def test_window_title_contains_version(qtbot):
    """タイトルにバージョン番号が含まれる"""
    win = MainWindow()
    qtbot.addWidget(win)
    assert "v0.5" in win.windowTitle()


def test_canvas_exists(qtbot):
    """キャンバスが生成されている"""
    win = MainWindow()
    qtbot.addWidget(win)
    assert win._canvas is not None


def test_grabcut_request_id_starts_at_zero(qtbot):
    """GrabCutリクエストIDが0から始まる"""
    win = MainWindow()
    qtbot.addWidget(win)
    assert win._grabcut_request_id == 0
