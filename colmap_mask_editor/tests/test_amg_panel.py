"""V0.8: 全画像自動分割パネル / コントローラ / 3タブ構成 / レビュー画面の GUI テスト。

Worker を起動せず、ウィジェットの構築・状態・シグナル・レビュー判断ロジックを検証する。
GUI は torch/sam2 を import しない。
"""

import numpy as np
import pytest

from ai import amg_manifest as M, amg_npz, amg_rle
from ai.amg_review_state import SegmentDecision
from ui.amg_batch_panel import AmgBatchPanel
from ui.amg_review_widget import AmgReviewWidget


# ----- バッチパネル -----

def test_panel_builds_and_defaults(qtbot):
    p = AmgBatchPanel()
    qtbot.addWidget(p)
    opts = p.options()
    assert opts["preset"] == "fast"
    assert opts["settings"]["points_per_side"] == 16  # 高速プリセット
    assert opts["scope"] in {s for _, s in __import__("ui.amg_batch_panel", fromlist=["SCOPE_ITEMS"]).SCOPE_ITEMS}


def test_panel_preset_switch(qtbot):
    p = AmgBatchPanel()
    qtbot.addWidget(p)
    # プリセットを標準へ
    idx = next(i for i in range(p._preset.count()) if p._preset.itemData(i) == "standard")
    p._preset.setCurrentIndex(idx)
    assert p.options()["settings"]["points_per_side"] == 32
    assert p.options()["preset"] == "standard"


def test_panel_custom_on_detail_edit(qtbot):
    p = AmgBatchPanel()
    qtbot.addWidget(p)
    p._pps.setValue(20)  # 個別変更
    assert p.options()["preset"] == "custom"


def test_panel_running_state_buttons(qtbot):
    p = AmgBatchPanel()
    qtbot.addWidget(p)
    p.set_running(True, paused=False, has_results=False)
    assert not p._btn_start.isEnabled()
    assert p._btn_pause.isEnabled()
    assert not p._btn_resume.isEnabled()
    assert p._btn_cancel.isEnabled()
    p.set_running(True, paused=True, has_results=False)
    assert p._btn_resume.isEnabled()
    p.set_running(False, paused=False, has_results=True)
    assert p._btn_start.isEnabled()
    assert p._btn_review.isEnabled()


def test_panel_signals(qtbot):
    p = AmgBatchPanel()
    qtbot.addWidget(p)
    with qtbot.waitSignal(p.start_requested, timeout=1000):
        p._btn_start.click()
    with qtbot.waitSignal(p.review_requested, timeout=1000):
        p._btn_review.setEnabled(True)
        p._btn_review.click()


# ----- 3タブ構成 (MainWindow) -----

def test_mainwindow_has_amg_tab_and_experimental_label(qtbot):
    from ui.main_window import MainWindow
    win = MainWindow()
    qtbot.addWidget(win)
    # AIセグメント内のトグルが4つ (単一画像/全画像自動分割/完全被覆リージョン/画像伝播)
    assert win._ai_view_single.text() == "単一画像"
    assert win._ai_view_amg.text() == "全画像自動分割"
    assert win._ai_view_partition.text() == "完全被覆リージョン"  # V0.9
    assert "実験的" in win._ai_view_prop.text()
    # スタックが4面 (V0.9 で完全被覆リージョンを追加)
    assert win._ai_stack.count() == 4
    assert hasattr(win, "_amg_panel")
    assert hasattr(win, "_amg_controller")
    assert hasattr(win, "_partition_panel")
    assert hasattr(win, "_partition_controller")


# ----- レビュー画面 -----

def _ann(m):
    h, w = m.shape
    ys, xs = np.where(m > 0)
    return {
        "segmentation": {"size": [h, w], "counts": amg_rle.encode_mask(m)},
        "area": int((m > 0).sum()),
        "bbox": [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
        "predicted_iou": 0.9, "stability_score": 0.95,
        "point_coords": [[float(xs.mean()), float(ys.mean())]], "crop_box": [0, 0, w, h],
    }


def _make_project(tmp_path):
    h, w = 24, 32
    big = np.zeros((h, w), np.uint8); big[2:20, 2:26] = 1
    small = np.zeros((h, w), np.uint8); small[8:13, 8:14] = 1
    arrays = amg_npz.build_segment_arrays([_ann(big), _ann(small)], h, w)
    key = "IMG_000.png"
    cache_dir = tmp_path / "segmentation_cache" / "images" / M.cache_id_for(key)
    cache_dir.mkdir(parents=True)
    src = tmp_path / "images" / key
    src.parent.mkdir(parents=True)
    import cv2
    img = np.zeros((h, w, 3), np.uint8); img[2:20, 2:26] = (200, 200, 200)
    ok, buf = cv2.imencode(".png", img); buf.tofile(str(src))
    sha = amg_npz.save_segments_npz(cache_dir / "segments.npz", arrays)
    man = M.build_image_manifest(
        image_key=key, source_path=str(src), width=w, height=h,
        model={"model_id": "m", "sam2_commit": "c", "checkpoint_fingerprint": "f"},
        generator=M.preset_settings("fast"), preset="fast",
        segment_count=2, segment_ids=arrays["segment_ids"].tolist(),
        segments_npz_sha256=sha, processing_time_sec=1.0,
        fingerprint={"file_size": 1, "mtime_ns": 2},
    )
    M.atomic_write_json(cache_dir / "manifest.json", man)
    items = [{"image_key": key, "source_path": str(src),
              "cache_id": M.cache_id_for(key), "status": "ready", "review_completed": False}]
    return items, key, cache_dir


def test_review_widget_builds_and_lists_segments(qtbot, tmp_path):
    items, key, cache_dir = _make_project(tmp_path)
    w = AmgReviewWidget(tmp_path, items)
    qtbot.addWidget(w)
    assert w._table.rowCount() == 2
    assert w._image_list.count() == 1


def test_review_decision_updates_manifest_only(qtbot, tmp_path):
    items, key, cache_dir = _make_project(tmp_path)
    w = AmgReviewWidget(tmp_path, items)
    qtbot.addWidget(w)
    npz_before = (cache_dir / "segments.npz").read_bytes()
    # segment_id 1 を KEEP, 2 を REMOVE
    w.set_decision_by_id(1, SegmentDecision.KEEP)
    w.set_decision_by_id(2, SegmentDecision.REMOVE)
    w._save_decisions()
    man = M.read_json(cache_dir / "manifest.json")
    assert man["review"]["decisions"]["1"] == "keep"
    assert man["review"]["decisions"]["2"] == "remove"
    # NPZ は不変
    assert (cache_dir / "segments.npz").read_bytes() == npz_before


def test_review_click_selects_smallest_candidate(qtbot, tmp_path):
    items, key, cache_dir = _make_project(tmp_path)
    w = AmgReviewWidget(tmp_path, items)
    qtbot.addWidget(w)
    # (10,10) は big と small の両方に含まれる -> 小さい候補(id2)が KEEP される
    w._on_canvas_clicked(10, 10, 1, False)
    man_decisions = w._decisions
    # small (id 2) が keep
    assert man_decisions["2"] == "keep"


def test_review_final_mask_signal(qtbot, tmp_path):
    items, key, cache_dir = _make_project(tmp_path)
    w = AmgReviewWidget(tmp_path, items)
    qtbot.addWidget(w)
    w.set_decision_by_id(1, SegmentDecision.KEEP)
    with qtbot.waitSignal(w.final_mask_requested, timeout=1000) as sig:
        w._emit_final()
    assert sig.args[0] == key


# ----- コントローラのエラー振り分け (画像単位 vs バッチ致命) -----

from PySide6.QtCore import QObject, Signal  # noqa: E402

from ai.amg_protocol import AmgErrorCode, AmgEvent, make_job_error, make_job_event  # noqa: E402
from ui.amg_controller import AmgController  # noqa: E402


class _FakePM(QObject):
    event_received = Signal(dict)
    error_received = Signal(dict)

    def __init__(self):
        super().__init__()
        self._rid = 0

    def send_command(self, *a, **k):
        self._rid += 1
        return self._rid


def _running_controller():
    pm = _FakePM()
    c = AmgController(pm)
    rid = c.start(project_root="r", images=[], settings={}, preset="fast", model={})
    pm.event_received.emit(make_job_event(AmgEvent.BATCH_STARTED, "amg-1", request_id=rid))
    assert c.is_running()
    return pm, c


def test_controller_per_image_error_does_not_fail_batch(qtbot):
    """image_key 付きエラーは image_failed のみ。バッチは停止しない。"""
    pm, c = _running_controller()
    failed, img_failed = [], []
    c.failed.connect(lambda code, msg: failed.append((code, msg)))
    c.image_failed.connect(lambda m: img_failed.append(m))
    pm.error_received.emit(make_job_error(
        AmgErrorCode.GENERATION_FAILED, "boom", job_id="amg-1", image_key="IMG_001.jpg"))
    assert img_failed and not failed
    assert c.is_running() is True


def test_controller_fatal_error_fails_batch(qtbot):
    """image_key 無しエラー (開始失敗 / バッチ致命) はバッチを停止させる。"""
    pm, c = _running_controller()
    failed = []
    c.failed.connect(lambda code, msg: failed.append((code, msg)))
    pm.error_received.emit(make_job_error(
        AmgErrorCode.GENERATION_FAILED, "fatal", job_id="amg-1"))
    assert failed and failed[0][0] == AmgErrorCode.GENERATION_FAILED
