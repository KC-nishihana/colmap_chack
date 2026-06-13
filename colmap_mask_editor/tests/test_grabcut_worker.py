"""
GrabCutWorkerのシグナル・キャンセル・例外処理テスト (v0.4B)

worker.run() を直接呼び出し（スレッドなし）でシグナルを検証する。
QtThread を使わない同期テストにより、タイムアウト・デッドロックを回避する。

V0.4B変更点:
  - WorkerはINITIAL/REFINEの2タスクをサポート
  - INITIALタスクは create_grabcut_session をパッチ
  - worker.result はGrabCutResult (後方互換), worker.session はGrabCutSession
"""

import inspect

import cv2
import numpy as np
import pytest

from core.grabcut_tool import GrabCutOptions, GrabCutResult, GrabCutSession
from core.grabcut_worker import GrabCutTaskType, GrabCutWorker


# ------------------------------------------------------------------ #
# テスト用ヘルパー
# ------------------------------------------------------------------ #

def make_small_image() -> np.ndarray:
    img = np.zeros((60, 80, 3), dtype=np.uint8)
    img[10:50, 10:70] = (180, 180, 180)
    return img


def make_dummy_session() -> GrabCutSession:
    """テスト用GrabCutSession。最小限のndarrayを持つ。"""
    h, w = 60, 80
    roi = (5, 5, 60, 45)
    ph, pw = 45, 60  # processing size
    mask_preview = np.zeros((h, w), dtype=np.uint8)
    mask_preview[15:45, 15:65] = 255
    label_mask = np.zeros((ph, pw), dtype=np.uint8)
    label_mask[5:40, 5:55] = 3  # GC_PR_FGD
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


def _make_refine_worker(request_id: int = 1) -> GrabCutWorker:
    return GrabCutWorker(
        request_id=request_id,
        task_type=GrabCutTaskType.REFINE,
        session=make_dummy_session(),
        hint_strokes=[],
        options=GrabCutOptions(iter_count=2),
    )


def _collect_signal(worker: GrabCutWorker, signal_name: str) -> list:
    """シグナルを受け取ったら results に追記するリストを返す。"""
    results: list = []
    signal = getattr(worker, signal_name)
    signal.connect(lambda *args: results.append(args))
    return results


# ------------------------------------------------------------------ #
# 正常処理テスト (INITIAL) - create_grabcut_session をパッチ
# ------------------------------------------------------------------ #

def test_finished_signal_emitted_on_success(qtbot, monkeypatch):
    """正常処理でfinishedシグナルが送出される"""
    dummy_session = make_dummy_session()
    monkeypatch.setattr(
        "core.grabcut_worker.create_grabcut_session", lambda *a, **k: dummy_session
    )
    worker = _make_worker(1)
    finished = _collect_signal(worker, "finished")

    worker.run()

    assert len(finished) == 1
    assert finished[0] == (1,)         # Signal(int) → request_id を含むタプル
    assert isinstance(worker.result, GrabCutResult)   # 後方互換のためGrabCutResult
    assert worker.session is dummy_session             # セッションも保存される


def test_result_is_grabcut_result_instance(qtbot, monkeypatch):
    """INITIALタスク: worker.result は GrabCutResult のインスタンス"""
    dummy_session = make_dummy_session()
    monkeypatch.setattr(
        "core.grabcut_worker.create_grabcut_session", lambda *a, **k: dummy_session
    )
    worker = _make_worker(2)
    worker.run()

    assert isinstance(worker.result, GrabCutResult)
    assert worker.result.mask is dummy_session.preview_mask


def test_session_stored_after_initial(qtbot, monkeypatch):
    """INITIALタスク: worker.session に GrabCutSession が格納される"""
    dummy_session = make_dummy_session()
    monkeypatch.setattr(
        "core.grabcut_worker.create_grabcut_session", lambda *a, **k: dummy_session
    )
    worker = _make_worker(3)
    worker.run()

    assert worker.session is dummy_session


# ------------------------------------------------------------------ #
# 正常処理テスト (REFINE) - refine_grabcut_session をパッチ
# ------------------------------------------------------------------ #

def test_refine_worker_finished_signal(qtbot, monkeypatch):
    """REFINEタスク: 正常処理でfinishedシグナルが送出される"""
    dummy_session = make_dummy_session()
    dummy_session_refined = make_dummy_session()
    dummy_session_refined = GrabCutSession(
        **{**dummy_session.__dict__, "refine_count": 1}
    )
    monkeypatch.setattr(
        "core.grabcut_worker.refine_grabcut_session",
        lambda *a, **k: dummy_session_refined,
    )
    worker = _make_refine_worker(10)
    finished = _collect_signal(worker, "finished")

    worker.run()

    assert len(finished) == 1
    assert worker.session is dummy_session_refined
    assert worker.result is dummy_session_refined


def test_refine_task_type_auto_detected():
    """sessionを渡すとREFINEタスクに自動判定される"""
    worker = GrabCutWorker(
        session=make_dummy_session(),
        hint_strokes=[],
    )
    assert worker._task_type == GrabCutTaskType.REFINE


def test_initial_task_type_default():
    """sessionなしはINITIALタスク"""
    worker = _make_worker()
    assert worker._task_type == GrabCutTaskType.INITIAL


def test_refine_requires_session(qtbot):
    """REFINEタスクでsession=Noneの場合はfailedシグナルを送出する"""
    worker = GrabCutWorker(
        request_id=99,
        task_type=GrabCutTaskType.REFINE,
        session=None,
        hint_strokes=[],
        options=GrabCutOptions(),
    )
    failed = _collect_signal(worker, "failed")

    worker.run()

    assert len(failed) == 1


# ------------------------------------------------------------------ #
# エラー処理テスト
# ------------------------------------------------------------------ #

def test_failed_signal_on_value_error(qtbot, monkeypatch):
    """ValueErrorでfailedシグナルが送出される"""
    monkeypatch.setattr(
        "core.grabcut_worker.create_grabcut_session",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("矩形が小さすぎます"))
    )
    worker = _make_worker(20)
    failed = _collect_signal(worker, "failed")

    worker.run()

    assert len(failed) == 1
    assert isinstance(failed[0][0], str)
    assert len(failed[0][0]) > 0


def test_failed_signal_on_cv2_error(qtbot, monkeypatch):
    """cv2.errorでfailedシグナルが送出される"""
    def raise_cv2_error(*a, **k):
        raise cv2.error("GrabCut失敗")

    monkeypatch.setattr("core.grabcut_worker.create_grabcut_session", raise_cv2_error)
    worker = _make_worker(21)
    failed = _collect_signal(worker, "failed")

    worker.run()

    assert len(failed) == 1
    msg = failed[0][0]
    assert "OpenCV" in msg or "GrabCut" in msg


def test_refine_failed_signal_on_cv2_error(qtbot, monkeypatch):
    """REFINEタスクでcv2.errorが発生した場合にfailedシグナルが送出される"""
    def raise_cv2_error(*a, **k):
        raise cv2.error("GrabCut再推定失敗")

    monkeypatch.setattr("core.grabcut_worker.refine_grabcut_session", raise_cv2_error)
    worker = _make_refine_worker(22)
    failed = _collect_signal(worker, "failed")

    worker.run()

    assert len(failed) == 1


# ------------------------------------------------------------------ #
# キャンセルテスト
# ------------------------------------------------------------------ #

def test_cancelled_signal_when_cancel_before_run(qtbot, monkeypatch):
    """キャンセルフラグが立っていると最初のチェックでcancelledを送出する"""
    dummy_session = make_dummy_session()
    monkeypatch.setattr(
        "core.grabcut_worker.create_grabcut_session", lambda *a, **k: dummy_session
    )
    worker = _make_worker(30)
    worker.request_cancel()
    cancelled = _collect_signal(worker, "cancelled")

    worker.run()

    assert len(cancelled) == 1


def test_cancelled_result_not_applied(qtbot, monkeypatch):
    """キャンセル済みの場合にfinishedが送出されない"""
    dummy_session = make_dummy_session()
    monkeypatch.setattr(
        "core.grabcut_worker.create_grabcut_session", lambda *a, **k: dummy_session
    )
    worker = _make_worker(31)
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
    dummy_session = make_dummy_session()
    monkeypatch.setattr(
        "core.grabcut_worker.create_grabcut_session", lambda *a, **k: dummy_session
    )
    worker = _make_worker(40)
    progress = _collect_signal(worker, "progress")

    worker.run()

    assert len(progress) >= 1


def test_refine_progress_signal_emitted(qtbot, monkeypatch):
    """REFINEタスクでもprogressシグナルが送出される"""
    dummy_session_refined = make_dummy_session()
    monkeypatch.setattr(
        "core.grabcut_worker.refine_grabcut_session",
        lambda *a, **k: dummy_session_refined,
    )
    worker = _make_refine_worker(41)
    progress = _collect_signal(worker, "progress")

    worker.run()

    assert len(progress) >= 1


# ------------------------------------------------------------------ #
# request_id テスト
# ------------------------------------------------------------------ #

def test_request_id_stored_correctly():
    worker = GrabCutWorker(make_small_image(), (5, 5, 60, 45), GrabCutOptions(), 42)
    assert worker.request_id == 42
