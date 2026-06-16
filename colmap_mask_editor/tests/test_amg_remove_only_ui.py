"""V0.10: REMOVE_ONLY レビュー画面 (AmgReviewWidget) の GUI テスト。

Worker を実起動して review_index を構築する。GUI は torch / sam2 を import しない。
"""

import sys

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QMessageBox

from ai import amg_manifest as M, amg_npz, amg_rle
from ai.amg_review_state import SegmentDecision
from ui.amg_review_widget import AmgReviewWidget


def _ann(m, iou=0.9, stab=0.95):
    h, w = m.shape
    ys, xs = np.where(m > 0)
    return {
        "segmentation": {"size": [h, w], "counts": amg_rle.encode_mask(m)},
        "area": int((m > 0).sum()),
        "bbox": [int(xs.min()), int(ys.min()),
                 int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
        "predicted_iou": float(iou), "stability_score": float(stab),
        "point_coords": [[float(xs.mean()), float(ys.mean())]], "crop_box": [0, 0, w, h],
    }


def _make_project(tmp_path, keys=("IMG_000.png",)):
    h, w = 40, 40
    items = []
    for key in keys:
        top = np.zeros((h, w), np.uint8); top[2:18, 2:38] = 1        # 上の帯 (端接触・独立)
        midb = np.zeros((h, w), np.uint8); midb[22:34, 4:16] = 1     # 下左ブロック B
        # C は B のほぼ複製 (1px だけ移動 -> IoU ~0.99)。面積は B と同じにして
        # ヒットテストの (面積昇順, index) で B が先頭に来る順序を保つ。
        midc = midb.copy(); midc[22, 4] = 0; midc[22, 16] = 1        # B の重複 (高 IoU)
        small = np.zeros((h, w), np.uint8); small[25:30, 6:11] = 1   # B 内の小 (B が covered する)
        arrays = amg_npz.build_segment_arrays(
            [_ann(top), _ann(midb), _ann(midc), _ann(small)], h, w)
        cache_dir = tmp_path / "segmentation_cache" / "images" / M.cache_id_for(key)
        cache_dir.mkdir(parents=True)
        src = tmp_path / "images" / key
        src.parent.mkdir(parents=True, exist_ok=True)
        import cv2
        img = np.full((h, w, 3), 120, np.uint8)
        ok, buf = cv2.imencode(".png", img); buf.tofile(str(src))
        sha = amg_npz.save_segments_npz(cache_dir / "segments.npz", arrays)
        man = M.build_image_manifest(
            image_key=key, source_path=str(src), width=w, height=h,
            model={"model_id": "m", "sam2_commit": "c", "checkpoint_fingerprint": "f"},
            generator=M.preset_settings("fast"), preset="fast",
            segment_count=len(arrays["segment_ids"]),
            segment_ids=arrays["segment_ids"].tolist(),
            segments_npz_sha256=sha, processing_time_sec=1.0,
            fingerprint={"file_size": 1, "mtime_ns": 2},
        )
        M.atomic_write_json(cache_dir / "manifest.json", man)
        items.append({"image_key": key, "source_path": str(src),
                      "cache_id": M.cache_id_for(key), "status": "ready",
                      "review_completed": False})
    return items, list(keys)


def _seg_id_at(w, x, y):
    from ai import amg_hit_test
    cands = amg_hit_test.candidates_at_point(w._npz, x, y)
    return w._id_by_seg_index[cands[0]]


def _press(w, key, mods=Qt.KeyboardModifier.NoModifier):
    ev = QKeyEvent(QKeyEvent.Type.KeyPress, key, mods)
    w.keyPressEvent(ev)


def test_remove_only_is_default(qtbot, tmp_path):
    items, keys = _make_project(tmp_path)
    w = AmgReviewWidget(tmp_path, items)
    qtbot.addWidget(w)
    assert w._workflow == "remove_only"
    assert w._workflow_combo.currentData() == "remove_only"
    # KEEP ボタンは非表示
    assert not w._btn_keep.isVisible() or w._btn_keep.isHidden()


def test_left_click_removes_right_click_releases(qtbot, tmp_path):
    items, keys = _make_project(tmp_path)
    w = AmgReviewWidget(tmp_path, items)
    qtbot.addWidget(w)
    sid = _seg_id_at(w, 15, 15)
    w._on_canvas_clicked(15, 15, 1, False)         # 左 -> REMOVE
    assert w._decisions[str(sid)] == "remove"
    w._on_canvas_clicked(15, 15, 2, False)         # 右 -> 解除
    assert w._decisions[str(sid)] == "unreviewed"
    w._on_canvas_clicked(15, 15, 1, True)          # Ctrl+左 -> 解除
    assert w._decisions[str(sid)] == "unreviewed"


def test_r_and_u_keys(qtbot, tmp_path):
    items, keys = _make_project(tmp_path)
    w = AmgReviewWidget(tmp_path, items, auto_advance=False, hide_covered=False)
    qtbot.addWidget(w)
    w._on_canvas_clicked(15, 15, 1, False)
    sid = _seg_id_at(w, 15, 15)
    _press(w, Qt.Key.Key_U)
    assert w._decisions[str(sid)] == "unreviewed"
    _press(w, Qt.Key.Key_R)
    assert w._decisions[str(sid)] == "remove"


def test_enter_removes_and_advances(qtbot, tmp_path):
    items, keys = _make_project(tmp_path)
    w = AmgReviewWidget(tmp_path, items, auto_advance=True, hide_covered=False)
    qtbot.addWidget(w)
    with qtbot.waitSignal(w.index_ready, timeout=5000):
        pass
    # 最初の可視候補を選択
    w._goto_relative(forward=True)
    first = w._current_seg_index
    sid = w._id_by_seg_index[first]
    _press(w, Qt.Key.Key_Return)
    assert w._decisions[str(sid)] == "remove"
    assert w._current_seg_index != first   # 次候補へ移動した


def test_multi_select_bulk_remove_and_undo(qtbot, tmp_path):
    items, keys = _make_project(tmp_path)
    w = AmgReviewWidget(tmp_path, items, representatives_only=False, hide_covered=False)
    qtbot.addWidget(w)
    # 複数行を選択
    w._table.selectAll()
    sel = w._selected_seg_indices()
    assert len(sel) >= 2
    w._bulk_remove_selected()
    for i in sel:
        assert w._decisions[str(w._id_by_seg_index[i])] == "remove"
    # 一括は 1 ステップで Undo
    w._undo_decision()
    for i in sel:
        assert w._decisions[str(w._id_by_seg_index[i])] != "remove"


def test_representatives_only_reduces_count(qtbot, tmp_path):
    items, keys = _make_project(tmp_path)
    w = AmgReviewWidget(tmp_path, items, representatives_only=True, hide_covered=False)
    qtbot.addWidget(w)
    with qtbot.waitSignal(w.index_ready, timeout=5000):
        pass
    reps_rows = w._table.rowCount()
    w._chk_reps_only.setChecked(False)   # 全候補表示
    all_rows = w._table.rowCount()
    assert all_rows == 4
    assert reps_rows < all_rows          # B,C が 1 グループに -> 代表のみで減る


def test_hide_covered_toggle(qtbot, tmp_path):
    items, keys = _make_project(tmp_path)
    w = AmgReviewWidget(tmp_path, items, representatives_only=False, hide_covered=True)
    qtbot.addWidget(w)
    with qtbot.waitSignal(w.index_ready, timeout=5000):
        pass
    # B を REMOVE -> B 内の small / 重複 C は covered になり隠れる
    b_sid = _seg_id_at(w, 14, 33)        # B のみの領域 (small 外)
    w.set_decision_by_id(b_sid, SegmentDecision.REMOVE)
    rows_hidden = w._table.rowCount()
    w._chk_hide_covered.setChecked(False)   # covered も表示
    rows_shown = w._table.rowCount()
    assert rows_shown > rows_hidden
    # 判断値は書き換えられていない (small は unreviewed のまま)
    small_sid = _seg_id_at(w, 8, 27)
    assert w._decisions[str(small_sid)] == "unreviewed"


def test_complete_does_not_require_all_reviewed(qtbot, tmp_path, monkeypatch):
    items, keys = _make_project(tmp_path)
    w = AmgReviewWidget(tmp_path, items)
    qtbot.addWidget(w)
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: QMessageBox.StandardButton.Yes)
    # 1 候補だけ REMOVE、残りは未確認のまま完了できる
    sid = _seg_id_at(w, 15, 15)
    w.set_decision_by_id(sid, SegmentDecision.REMOVE)
    w._complete_review()
    man = M.read_json(w._cur_manifest_path)
    assert man["review"]["completed"] is True
    assert man["review"]["workflow"] == "remove_only"
    # decisions は remove のみ保存 (未確認の大量保存をしない)
    assert all(v == "remove" for v in man["review"]["decisions"].values())


def test_auto_next_image_after_complete(qtbot, tmp_path, monkeypatch):
    items, keys = _make_project(tmp_path, keys=("A.png", "B.png"))
    w = AmgReviewWidget(tmp_path, items, auto_next_image=True)
    qtbot.addWidget(w)
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: QMessageBox.StandardButton.Yes)
    assert w._image_list.currentRow() == 0
    w._complete_review()
    assert w._image_list.currentRow() == 1   # 次の未完了画像へ移動


def test_final_mask_mode_fixed_in_remove_only(qtbot, tmp_path):
    items, keys = _make_project(tmp_path)
    w = AmgReviewWidget(tmp_path, items)
    qtbot.addWidget(w)
    assert w._final_mode.currentData() == "exclude_remove"
    assert not w._final_mode.isEnabled()


def test_gui_does_not_import_torch_or_sam2():
    import ui.amg_review_widget  # noqa: F401
    assert "torch" not in sys.modules
    assert "sam2" not in sys.modules
