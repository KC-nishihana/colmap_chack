"""
GrabCutWorkerのシグナル・キャンセル・例外処理テスト (v0.4A.1)

worker.run() を直接呼び出し（スレッドなし）でシグナルを検証する。
QtThread を使わない同期テストにより、タイムアウト・デッドロックを回避する。
"""

import inspect

import cv2
import numpy as np
import pytest

from core.grabcut_tool import GrabCutOptions, GrabCutResult
from core.grabcut_worker import GrabCutWorker


# ------------------------------------------------------------------ #
# テスト用ヘルパー
# ------------------------------------------------------------------ #

def make_small_image() -> np.ndarray:
    img = np.zeros((60, 80, 3), dtype=np.uint8)
    img[10:50, 10:70] = (180, 180, 180)
    return img


def make_dummy_result() -> GrabCutResult:
    mask = np.zeros((60, 80), dtype=np.uint8)
    mask[15:45, 15:65] = 255
    return GrabCutResult(
        mask=mask,
        original_size=(80, 60),
        roi=(0, 0, 80, 60),
        processing_size=(80, 60),
        scale=1.0,
        processing_time_sec=0.001,
        was_downscaled=False,
    )


def _make_worker(request_id: int = 1) -> GrabCutWorker:
    return GrabCutWorker(make_small_image(), (5, 5, 60, 45), GrabCutOptions(), request_id)


def _collect_signal(worker: GrabCutWorker, signal_name: str) -> list:
    """シグナルを受け取ったら results に追記するリストを返す。"""
    results: list = []
    signal = getattr(worker, signal_name)
    signal.connect(lambda *args: results.append(args))
    return results


# ------------------------------------------------------------------ #
# 正常処理テスト (スレッドなし同期呼び出し)
# ------------------------------------------------------------------ #

def test_finished_signal_emitted_on_success(qtbot, monkeypatch):
    """正常処理でfinishedシグナルが送出される"""
    dummy = make_dummy_result()
    monkeypatch.setattr(
        "core.grabcut_worker.run_grabcut_optimized", lambda *a, **k: dummy
    )
    worker = _make_worker(1)
    finished = _collect_signal(worker, "finished")

    worker.run()

    assert len(finished) == 1
    assert isinstance(finished[0][0], GrabCutResult)


# ------------------------------------------------------------------ #
# エラー処理テスト
# ------------------------------------------------------------------ #

def test_failed_signal_on_value_error(qtbot, monkeypatch):
    """ValueErrorでfailedシグナルが送出される"""
    monkeypatch.setattr(
        "core.grabcut_worker.run_grabcut_optimized",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("矩形が小さすぎます"))
    )
    worker = _make_worker(2)
    failed = _collect_signal(worker, "failed")

    worker.run()

    assert len(failed) == 1
    assert isinstance(failed[0][0], str)
    assert len(failed[0][0]) > 0


def test_failed_signal_on_cv2_error(qtbot, monkeypatch):
    """cv2.errorでfailedシグナルが送出される"""
    def raise_cv2_error(*a, **k):
        raise cv2.error("GrabCut失敗")

    monkeypatch.setattr("core.grabcut_worker.run_grabcut_optimized", raise_cv2_error)
    worker = _make_worker(3)
    failed = _collect_signal(worker, "failed")

    worker.run()

    assert len(failed) == 1
    msg = failed[0][0]
    assert "OpenCV" in msg or "GrabCut" in msg


# ------------------------------------------------------------------ #
# キャンセルテスト
# ------------------------------------------------------------------ #

def test_cancelled_signal_when_cancel_before_run(qtbot, monkeypatch):
    """キャンセルフラグが立っていると最初のチェックでcancelledを送出する"""
    dummy = make_dummy_result()
    monkeypatch.setattr(
        "core.grabcut_worker.run_grabcut_optimized", lambda *a, **k: dummy
    )
    worker = _make_worker(4)
    worker.request_cancel()
    cancelled = _collect_signal(worker, "cancelled")

    worker.run()

    assert len(cancelled) == 1


def test_cancelled_result_not_applied(qtbot, monkeypatch):
    """キャンセル済みの場合にfinishedが送出されない"""
    dummy = make_dummy_result()
    monkeypatch.setattr(
        "core.grabcut_worker.run_grabcut_optimized", lambda *a, **k: dummy
    )
    worker = _make_worker(5)
    worker.request_cancel()
    finished = _collect_signal(worker, "finished")

    worker.run()

    assert len(finished) == 0


# ------------------------------------------------------------------ #
# Qtウィジェット非操作テスト
# ------------------------------------------------------------------ #

def test_worker_does_not_import_qt_widgets():
    """GrabCutWorkerがQtWidgetsをインポートしていないことを確認"""
    import core.grabcut_worker as module
    source = inspect.getsource(module)
    assert "QtWidgets" not in source, (
        "GrabCutWorkerはQtWidgetsをインポートしてはいけません"
    )


def test_worker_does_not_subclass_widget():
    """GrabCutWorkerがQWidgetのサブクラスでないことを確認"""
    try:
        from PySide6.QtWidgets import QWidget
        assert not issubclass(GrabCutWorker, QWidget)
    except ImportError:
        pass


# ------------------------------------------------------------------ #
# progress シグナルテスト
# ------------------------------------------------------------------ #

def test_progress_signal_emitted(qtbot, monkeypatch):
    """処理中にprogressシグナルが送出される"""
    dummy = make_dummy_result()
    monkeypatch.setattr(
        "core.grabcut_worker.run_grabcut_optimized", lambda *a, **k: dummy
    )
    worker = _make_worker(6)
    progress = _collect_signal(worker, "progress")

    worker.run()

    assert len(progress) >= 1


# ------------------------------------------------------------------ #
# request_id テスト
# ------------------------------------------------------------------ #

def test_request_id_stored_correctly():
    worker = GrabCutWorker(make_small_image(), (5, 5, 60, 45), GrabCutOptions(), 42)
    assert worker.request_id == 42
