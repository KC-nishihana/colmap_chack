"""
GrabCut QThread統合テスト (v0.5.1)

実際に QThread へ Worker を移動して動作させる統合テスト。
重い GrabCut 処理はモンキーパッチしてダミー処理を使う。
"""

import time

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QThread

from core.grabcut_tool import GrabCutOptions, GrabCutResult, GrabCutSession
from core.grabcut_worker import GrabCutTaskType, GrabCutWorker


# ------------------------------------------------------------------ #
# ヘルパー
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


def make_dummy_session() -> GrabCutSession:
    h, w = 60, 80
    roi = (5, 5, 60, 45)
    ph, pw = 45, 60
    mask_preview = np.zeros((h, w), dtype=np.uint8)
    mask_preview[15:45, 15:65] = 255
    label_mask = np.zeros((ph, pw), dtype=np.uint8)
    return GrabCutSession(
        original_size=(w, h),
        original_rect=(5, 5, 60, 45),
        roi=roi,
        processing_size=(pw, ph),
        scale=1.0,
        was_downscaled=False,
        roi_image_bgr=np.zeros((ph, pw, 3), dtype=np.uint8),
        base_label_mask=label_mask.copy(),
        label_mask=label_mask.copy(),
        bgd_model=np.zeros((1, 65), dtype=np.float64),
        fgd_model=np.zeros((1, 65), dtype=np.float64),
        preview_mask=mask_preview.copy(),
        processing_time_sec=0.001,
        refine_count=0,
    )


def run_worker_in_thread(worker: GrabCutWorker, qtbot, timeout_ms: int = 5000):
    """Workerを QThread で実行し、finished/failed/cancelled まで待機する。"""
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)

    # Worker終了でスレッドを停止
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    worker.cancelled.connect(thread.quit)
    thread.finished.connect(thread.deleteLater)

    signals = []
    worker.finished.connect(lambda rid: signals.append(("finished", rid)))
    worker.failed.connect(lambda msg, rid: signals.append(("failed", msg, rid)))
    worker.cancelled.connect(lambda rid: signals.append(("cancelled", rid)))

    thread.start()

    # スレッドが終了するまで待機
    deadline = time.monotonic() + timeout_ms / 1000.0
    while thread.isRunning() and time.monotonic() < deadline:
        QCoreApplication.processEvents()
        time.sleep(0.01)

    # スレッド終了後のキューイングシグナルを配信
    QCoreApplication.processEvents()

    return signals, thread


# ------------------------------------------------------------------ #
# INITIAL タスク (モンキーパッチあり)
# ------------------------------------------------------------------ #

def _make_patched_initial(thread_ids=None):
    """正常完了する _run_initial パッチを生成する。finished を emit する。"""
    def patched(self):
        if thread_ids is not None:
            import threading
            thread_ids.append(threading.get_ident())
        self.result = make_dummy_result()
        self.session = make_dummy_session()
        self.finished.emit(self._request_id)
    return patched


def test_worker_runs_in_separate_thread(qtbot, monkeypatch):
    """Worker が別スレッドで実行される"""
    thread_ids: list = []

    monkeypatch.setattr(GrabCutWorker, "_run_initial", _make_patched_initial(thread_ids))

    import threading
    main_thread_id = threading.get_ident()

    worker = GrabCutWorker(
        make_small_image(), (5, 5, 60, 45), GrabCutOptions(), request_id=1
    )
    signals, thread = run_worker_in_thread(worker, qtbot)

    assert len(signals) == 1
    assert signals[0][0] == "finished"
    if thread_ids:
        assert thread_ids[0] != main_thread_id


def test_finished_signal_triggers_thread_quit(qtbot, monkeypatch):
    """finished シグナル後にスレッドが終了する"""
    monkeypatch.setattr(GrabCutWorker, "_run_initial", _make_patched_initial())

    worker = GrabCutWorker(
        make_small_image(), (5, 5, 60, 45), GrabCutOptions(), request_id=1
    )
    signals, thread = run_worker_in_thread(worker, qtbot)

    assert any(s[0] == "finished" for s in signals)
    assert not thread.isRunning()


def test_failed_signal_triggers_thread_quit(qtbot, monkeypatch):
    """failed シグナル後にスレッドが終了する"""
    def raise_error(self):
        raise ValueError("ダミーエラー")

    monkeypatch.setattr(GrabCutWorker, "_run_initial", raise_error)

    worker = GrabCutWorker(
        make_small_image(), (5, 5, 60, 45), GrabCutOptions(), request_id=1
    )
    signals, thread = run_worker_in_thread(worker, qtbot)

    assert any(s[0] == "failed" for s in signals)
    assert not thread.isRunning()


def test_cancelled_signal_triggers_thread_quit(qtbot, monkeypatch):
    """cancelled シグナル後にスレッドが終了する"""
    def emit_cancelled(self):
        self.cancelled.emit(self._request_id)

    monkeypatch.setattr(GrabCutWorker, "_run_initial", emit_cancelled)

    worker = GrabCutWorker(
        make_small_image(), (5, 5, 60, 45), GrabCutOptions(), request_id=1
    )
    signals, thread = run_worker_in_thread(worker, qtbot)

    assert any(s[0] == "cancelled" for s in signals)
    assert not thread.isRunning()


# ------------------------------------------------------------------ #
# 古い request_id の結果を無視
# ------------------------------------------------------------------ #

def test_old_request_id_result_not_applied(qtbot, monkeypatch):
    """古い request_id の finished は MainWindow で無視される"""
    from ui.main_window import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)

    applied = [False]
    original_on_gc_finished = win._on_grabcut_finished

    def patched_on_gc_finished(result, session, request_id):
        applied[0] = True
        original_on_gc_finished(result, session, request_id)

    monkeypatch.setattr(win, "_on_grabcut_finished", patched_on_gc_finished)

    # request_id を 5 に設定
    win._grabcut_request_id = 5

    # 古い request_id=1 の finished を手動で送信
    win._grabcut_task_is_refine = False
    win._on_worker_finished(request_id=1)

    # 古いIDなので適用されない
    assert applied[0] is False


# ------------------------------------------------------------------ #
# REFINE タスク
# ------------------------------------------------------------------ #

def test_refine_worker_finished_triggers_thread_quit(qtbot, monkeypatch):
    """REFINE Worker の finished でスレッドが終了する"""
    session = make_dummy_session()

    def patched_run_refine(self):
        new_session = make_dummy_session()
        self.result = new_session
        self.session = new_session
        self.finished.emit(self._request_id)

    monkeypatch.setattr(GrabCutWorker, "_run_refine", patched_run_refine)

    worker = GrabCutWorker(
        request_id=1,
        task_type=GrabCutTaskType.REFINE,
        session=session,
        hint_strokes=[],
        options=GrabCutOptions(iter_count=1),
    )
    signals, thread = run_worker_in_thread(worker, qtbot)

    assert any(s[0] == "finished" for s in signals)
    assert not thread.isRunning()


# ------------------------------------------------------------------ #
# Worker/Thread 参照の安全な解放
# ------------------------------------------------------------------ #

def test_worker_reference_cleared_after_thread_finish(qtbot, monkeypatch):
    """Worker終了後に MainWindow の参照がクリアされる"""
    from ui.main_window import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)

    monkeypatch.setattr(GrabCutWorker, "_run_initial", _make_patched_initial())
    monkeypatch.setattr("ui.main_window.QMessageBox.warning", lambda *a, **k: None)

    # プレビュー適用をスキップ
    monkeypatch.setattr(win._canvas, "set_grabcut_preview", lambda *a: None)

    worker = GrabCutWorker(
        make_small_image(), (5, 5, 60, 45), GrabCutOptions(), request_id=1
    )
    win._grabcut_request_id = 1
    win._grabcut_task_is_refine = False
    win._start_worker(worker)

    deadline = time.monotonic() + 5.0
    while win._grabcut_worker is not None and time.monotonic() < deadline:
        QCoreApplication.processEvents()
        time.sleep(0.01)

    assert win._grabcut_worker is None
    assert win._grabcut_thread is None
