"""
GrabCutWorker REFINEタスクのユニットテスト (v0.4B)

cv2.grabCutをモックしてGC_INIT_WITH_MASKでの再推定動作を検証する。
"""

import cv2
import numpy as np
import pytest

from core.grabcut_tool import (
    GrabCutHintLabel,
    GrabCutOptions,
    GrabCutSession,
    HintStroke,
)
from core.grabcut_worker import GrabCutTaskType, GrabCutWorker


# ------------------------------------------------------------------ #
# テスト用ヘルパー
# ------------------------------------------------------------------ #

def make_image(h: int = 80, w: int = 100) -> np.ndarray:
    """GrabCutが前景を検出できる十分なコントラストを持つ画像。"""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = (30, 20, 10)
    cy, cx = h // 2, w // 2
    r = min(h, w) // 4
    cv2.circle(img, (cx, cy), r, (60, 120, 200), -1)
    return img


def make_session(h: int = 80, w: int = 100) -> GrabCutSession:
    """テスト用GrabCutSession。縮小なし。"""
    from core.grabcut_tool import create_grabcut_session
    img = make_image(h, w)
    cy, cx = h // 2, w // 2
    r = min(h, w) // 4
    rect = (cx - r, cy - r, r * 2, r * 2)
    opts = GrabCutOptions(iter_count=2, use_downscale=False)
    return create_grabcut_session(img, rect, opts)


def make_refine_worker(session: GrabCutSession, strokes=None, iter_count: int = 2) -> GrabCutWorker:
    if strokes is None:
        strokes = []
    return GrabCutWorker(
        request_id=1,
        task_type=GrabCutTaskType.REFINE,
        session=session,
        hint_strokes=strokes,
        options=GrabCutOptions(iter_count=iter_count),
    )


def _collect_signal(worker: GrabCutWorker, signal_name: str) -> list:
    results: list = []
    signal = getattr(worker, signal_name)
    signal.connect(lambda *args: results.append(args))
    return results


# ------------------------------------------------------------------ #
# GC_INIT_WITH_MASK が使われていること
# ------------------------------------------------------------------ #

def test_refine_uses_gc_init_with_mask(qtbot, monkeypatch):
    """再推定時にcv2.grabCutがGC_INIT_WITH_MASKで呼ばれる"""
    session = make_session()
    calls = []

    def mock_grabcut(img, mask, rect, bgd, fgd, n, flags):
        calls.append({"rect": rect, "flags": flags})
        # 前景ピクセルを設定して正常終了をシミュレート
        mask[:] = cv2.GC_PR_FGD

    monkeypatch.setattr(cv2, "grabCut", mock_grabcut)

    worker = make_refine_worker(session)
    finished = _collect_signal(worker, "finished")
    worker.run()

    assert len(calls) == 1
    assert calls[0]["flags"] == cv2.GC_INIT_WITH_MASK


def test_refine_passes_rect_none(qtbot, monkeypatch):
    """再推定時にrect=Noneでcv2.grabCutが呼ばれる"""
    session = make_session()
    calls = []

    def mock_grabcut(img, mask, rect, bgd, fgd, n, flags):
        calls.append({"rect": rect})
        mask[:] = cv2.GC_PR_FGD

    monkeypatch.setattr(cv2, "grabCut", mock_grabcut)

    worker = make_refine_worker(session)
    worker.run()

    assert calls[0]["rect"] is None


# ------------------------------------------------------------------ #
# ヒントストロークの反映
# ------------------------------------------------------------------ #

def test_fg_hints_in_refine(qtbot, monkeypatch):
    """FGヒントがGC_FGDとしてマスクに含まれる"""
    session = make_session()
    roi_x, roi_y, roi_w, roi_h = session.roi
    cx = roi_x + roi_w // 2
    cy = roi_y + roi_h // 2
    strokes = [HintStroke(label=GrabCutHintLabel.FOREGROUND, points=[(cx, cy)], radius=5)]

    received_masks = []

    def mock_grabcut(img, mask, rect, bgd, fgd, n, flags):
        received_masks.append(mask.copy())
        mask[:] = cv2.GC_PR_FGD

    monkeypatch.setattr(cv2, "grabCut", mock_grabcut)

    worker = make_refine_worker(session, strokes=strokes)
    worker.run()

    assert len(received_masks) == 1
    # FGヒント点付近がGC_FGDになっていること
    px = int(round((cx - roi_x) * session.scale))
    py = int(round((cy - roi_y) * session.scale))
    ph, pw = received_masks[0].shape
    px = max(0, min(px, pw - 1))
    py = max(0, min(py, ph - 1))
    assert received_masks[0][py, px] == cv2.GC_FGD


def test_bg_hints_in_refine(qtbot, monkeypatch):
    """BGヒントがGC_BGDとしてマスクに含まれる"""
    session = make_session()
    roi_x, roi_y, roi_w, roi_h = session.roi
    bx = roi_x + 3
    by = roi_y + 3
    strokes = [HintStroke(label=GrabCutHintLabel.BACKGROUND, points=[(bx, by)], radius=3)]

    received_masks = []

    def mock_grabcut(img, mask, rect, bgd, fgd, n, flags):
        received_masks.append(mask.copy())
        mask[:] = cv2.GC_PR_FGD

    monkeypatch.setattr(cv2, "grabCut", mock_grabcut)

    worker = make_refine_worker(session, strokes=strokes)
    worker.run()

    assert len(received_masks) == 1
    px = int(round((bx - roi_x) * session.scale))
    py = int(round((by - roi_y) * session.scale))
    ph, pw = received_masks[0].shape
    px = max(0, min(px, pw - 1))
    py = max(0, min(py, ph - 1))
    assert received_masks[0][py, px] == cv2.GC_BGD


# ------------------------------------------------------------------ #
# 出力形状・値の検証
# ------------------------------------------------------------------ #

def test_preview_mask_original_resolution(qtbot, monkeypatch):
    """再推定後のpreview_maskは元画像解像度"""
    session = make_session(60, 80)

    def mock_grabcut(img, mask, rect, bgd, fgd, n, flags):
        mask[:] = cv2.GC_PR_FGD

    monkeypatch.setattr(cv2, "grabCut", mock_grabcut)

    worker = make_refine_worker(session)
    finished = _collect_signal(worker, "finished")
    worker.run()

    assert len(finished) == 1
    assert worker.session is not None
    assert worker.session.preview_mask.shape == (60, 80)


def test_preview_mask_values_0_255(qtbot, monkeypatch):
    """再推定後のpreview_maskの値は0または255のみ"""
    session = make_session()

    def mock_grabcut(img, mask, rect, bgd, fgd, n, flags):
        mask[:] = cv2.GC_PR_FGD

    monkeypatch.setattr(cv2, "grabCut", mock_grabcut)

    worker = make_refine_worker(session)
    worker.run()

    unique = set(np.unique(worker.session.preview_mask).tolist())
    assert unique.issubset({0, 255})


def test_refine_count_incremented(qtbot, monkeypatch):
    """再推定後のrefine_countが1増える"""
    session = make_session()
    original_count = session.refine_count

    def mock_grabcut(img, mask, rect, bgd, fgd, n, flags):
        mask[:] = cv2.GC_PR_FGD

    monkeypatch.setattr(cv2, "grabCut", mock_grabcut)

    worker = make_refine_worker(session)
    worker.run()

    assert worker.session.refine_count == original_count + 1


def test_original_session_not_modified(qtbot, monkeypatch):
    """再推定は入力Sessionを破壊しない"""
    session = make_session()
    orig_base = session.base_label_mask.copy()
    orig_label = session.label_mask.copy()
    orig_refine_count = session.refine_count

    def mock_grabcut(img, mask, rect, bgd, fgd, n, flags):
        mask[:] = cv2.GC_PR_FGD

    monkeypatch.setattr(cv2, "grabCut", mock_grabcut)

    worker = make_refine_worker(session)
    worker.run()

    np.testing.assert_array_equal(session.base_label_mask, orig_base)
    np.testing.assert_array_equal(session.label_mask, orig_label)
    assert session.refine_count == orig_refine_count


# ------------------------------------------------------------------ #
# エラーケース
# ------------------------------------------------------------------ #

def test_fg_count_zero_raises_failed(qtbot, monkeypatch):
    """再推定後に前景ピクセルが0の場合はfailedシグナルが送出される"""
    session = make_session()

    def mock_grabcut(img, mask, rect, bgd, fgd, n, flags):
        mask[:] = cv2.GC_BGD  # 全部背景 → 前景0

    monkeypatch.setattr(cv2, "grabCut", mock_grabcut)

    worker = make_refine_worker(session)
    failed = _collect_signal(worker, "failed")
    worker.run()

    assert len(failed) == 1
    assert isinstance(failed[0][0], str)


def test_cv2_error_in_refine_emits_failed(qtbot, monkeypatch):
    """refine中にcv2.errorが発生した場合はfailedシグナルが送出される"""
    session = make_session()

    def raise_cv2_error(*a, **k):
        raise cv2.error("テストエラー")

    monkeypatch.setattr(cv2, "grabCut", raise_cv2_error)

    worker = make_refine_worker(session)
    failed = _collect_signal(worker, "failed")
    worker.run()

    assert len(failed) == 1


def test_session_none_emits_failed(qtbot):
    """session=Noneの場合はfailedシグナルが送出される"""
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
# 新セッション vs 入力セッション
# ------------------------------------------------------------------ #

def test_refine_returns_new_session(qtbot, monkeypatch):
    """再推定後のworker.sessionは入力Sessionとは別オブジェクト"""
    session = make_session()

    def mock_grabcut(img, mask, rect, bgd, fgd, n, flags):
        mask[:] = cv2.GC_PR_FGD

    monkeypatch.setattr(cv2, "grabCut", mock_grabcut)

    worker = make_refine_worker(session)
    worker.run()

    assert worker.session is not session
    assert isinstance(worker.session, GrabCutSession)


def test_result_equals_session_for_refine(qtbot, monkeypatch):
    """REFINEタスクでworker.result is worker.session"""
    session = make_session()

    def mock_grabcut(img, mask, rect, bgd, fgd, n, flags):
        mask[:] = cv2.GC_PR_FGD

    monkeypatch.setattr(cv2, "grabCut", mock_grabcut)

    worker = make_refine_worker(session)
    worker.run()

    assert worker.result is worker.session
