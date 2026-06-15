"""
V0.8 GUI: AMG 解析結果のレビュー画面 (QDialog)。

左: 画像一覧 (状態アイコン)  中央: 元画像 + セグメントオーバーレイ
右: セグメント一覧 / フィルタ / レビュー完了 / 最終マスク生成

クリック: 左=KEEP / 右=REMOVE / Ctrl+左=未確認。重複候補は Tab/Shift+Tab で切替。
判断は manifest.json の decisions のみ原子更新 (NPZ は書き換えない)。最終マスク生成は
MainWindow へシグナルで依頼する (実際の合成・保存は core.amg_apply_worker)。

torch / sam2 は import しない (numpy / cv2 / amg_* のみ)。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ai import amg_hit_test, amg_manifest, amg_mask_composer, amg_npz
from ai.amg_review_state import (
    SegmentDecision,
    count_decisions,
    is_review_complete,
    normalize_decisions,
)
from core.mask_io import imread_jp

_log = logging.getLogger(__name__)

# 画像一覧の状態アイコン
_STATUS_ICON = {
    "unprocessed": "○ 未処理",
    "ready": "✓ 解析済み",
    "partial": "△ 一部確認",
    "completed": "● 完了",
    "failed": "! 失敗",
    "stale": "↻ 古い",
    "corrupt": "✗ 破損",
}

# 候補色 (RGBA)
_COL_UNREVIEWED = (255, 230, 0)
_COL_KEEP = (0, 220, 60)
_COL_REMOVE = (235, 40, 40)
_COL_CURRENT = (0, 220, 255)

_FINAL_MODE_ITEMS = [
    ("不要領域を除外", amg_mask_composer.MODE_EXCLUDE_REMOVE),
    ("必要領域のみ", amg_mask_composer.MODE_KEEP_ONLY),
    ("現在マスクへ追加・除外", amg_mask_composer.MODE_ADD_REMOVE),
]


class _ReviewCanvas(QWidget):
    """元画像 + オーバーレイ表示。クリックを画像座標へ変換して emit する。"""

    clicked = Signal(int, int, int, bool)   # x, y, button(1=left,2=right), ctrl
    hovered = Signal(int, int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(480, 360)
        self.setMouseTracking(True)
        self._base_qimg: Optional[QImage] = None
        self._overlay_qimg: Optional[QImage] = None
        self._img_w = 0
        self._img_h = 0
        self._scale = 1.0
        self._off = (0, 0)

    def set_image(self, rgb: np.ndarray) -> None:
        self._img_h, self._img_w = rgb.shape[:2]
        buf = np.ascontiguousarray(rgb)
        self._base_qimg = QImage(buf.data, self._img_w, self._img_h,
                                 3 * self._img_w, QImage.Format.Format_RGB888).copy()
        self._overlay_qimg = None
        self.update()

    def set_overlay(self, rgba: Optional[np.ndarray]) -> None:
        if rgba is None:
            self._overlay_qimg = None
        else:
            buf = np.ascontiguousarray(rgba)
            self._overlay_qimg = QImage(buf.data, buf.shape[1], buf.shape[0],
                                        4 * buf.shape[1], QImage.Format.Format_RGBA8888).copy()
        self.update()

    def _recompute_transform(self) -> None:
        if self._img_w == 0 or self._img_h == 0:
            return
        sw, sh = self.width(), self.height()
        self._scale = min(sw / self._img_w, sh / self._img_h)
        dw, dh = self._img_w * self._scale, self._img_h * self._scale
        self._off = ((sw - dw) / 2, (sh - dh) / 2)

    def paintEvent(self, _event) -> None:
        from PySide6.QtGui import QPainter
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.GlobalColor.black)
        if self._base_qimg is None:
            return
        self._recompute_transform()
        ox, oy = self._off
        dw = int(self._img_w * self._scale)
        dh = int(self._img_h * self._scale)
        target = self._base_qimg.scaled(dw, dh, Qt.AspectRatioMode.KeepAspectRatio,
                                        Qt.TransformationMode.SmoothTransformation)
        p.drawImage(int(ox), int(oy), target)
        if self._overlay_qimg is not None:
            ov = self._overlay_qimg.scaled(dw, dh, Qt.AspectRatioMode.KeepAspectRatio,
                                           Qt.TransformationMode.SmoothTransformation)
            p.drawImage(int(ox), int(oy), ov)

    def _to_image_coords(self, wx: float, wy: float) -> Optional[tuple[int, int]]:
        if self._img_w == 0 or self._scale <= 0:
            return None
        ox, oy = self._off
        x = int((wx - ox) / self._scale)
        y = int((wy - oy) / self._scale)
        if 0 <= x < self._img_w and 0 <= y < self._img_h:
            return x, y
        return None

    def mousePressEvent(self, e) -> None:
        pos = self._to_image_coords(e.position().x(), e.position().y())
        if pos is None:
            return
        ctrl = bool(e.modifiers() & Qt.KeyboardModifier.ControlModifier)
        btn = 1 if e.button() == Qt.MouseButton.LeftButton else 2
        self.clicked.emit(pos[0], pos[1], btn, ctrl)

    def mouseMoveEvent(self, e) -> None:
        pos = self._to_image_coords(e.position().x(), e.position().y())
        if pos is not None:
            self.hovered.emit(pos[0], pos[1])


class AmgReviewWidget(QDialog):
    final_mask_requested = Signal(str, str)   # image_key, mode (現在画像)
    final_mask_batch_requested = Signal(list, str)  # image_keys, mode

    def __init__(self, project_root, image_items: list[dict], parent=None,
                 decode_cache_size: int = 12) -> None:
        super().__init__(parent)
        self.setWindowTitle("AMG レビュー — 必要 / 不要 / 未確認")
        self.resize(1200, 760)
        self._root = Path(project_root)
        self._items = list(image_items)
        self._decode_cache_size = int(decode_cache_size)

        # 現在画像の状態
        self._cur_key: Optional[str] = None
        self._cur_manifest_path: Optional[Path] = None
        self._npz = None
        self._cache: Optional[amg_hit_test.MaskDecodeCache] = None
        self._decisions: dict[str, str] = {}
        self._seg_index_by_id: dict[int, int] = {}
        self._id_by_seg_index: dict[int, int] = {}
        self._current_seg_index: Optional[int] = None
        self._candidates: list[int] = []
        self._base_rgb: Optional[np.ndarray] = None

        self._build_ui()
        self._populate_image_list()
        if self._items:
            self._image_list.setCurrentRow(0)

    # ------------------------------------------------------------------ #
    # UI 構築
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # 左: 画像一覧
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.addWidget(QLabel("画像一覧"))
        self._image_list = QListWidget()
        self._image_list.currentRowChanged.connect(self._on_image_changed)
        lv.addWidget(self._image_list)
        splitter.addWidget(left)

        # 中央: 画像 + オーバーレイ
        center = QWidget()
        cv = QVBoxLayout(center)
        self._canvas = _ReviewCanvas()
        self._canvas.clicked.connect(self._on_canvas_clicked)
        cv.addWidget(self._canvas, 1)
        ctl = QHBoxLayout()
        ctl.addWidget(QLabel("透明度"))
        self._opacity = QSlider(Qt.Orientation.Horizontal)
        self._opacity.setRange(0, 100)
        self._opacity.setValue(50)
        self._opacity.valueChanged.connect(lambda _v: self._refresh_overlay())
        ctl.addWidget(self._opacity)
        self._hint = QLabel("左=必要 / 右=不要 / Ctrl+左=未確認 / Tab=重複候補切替")
        self._hint.setStyleSheet("color:#9cf;")
        ctl.addWidget(self._hint, 1)
        cv.addLayout(ctl)
        splitter.addWidget(center)

        # 右: セグメント一覧 + フィルタ + 操作
        right = QWidget()
        rv = QVBoxLayout(right)
        self._counts_label = QLabel("KEEP 0 / REMOVE 0 / 未確認 0")
        rv.addWidget(self._counts_label)

        filt = QHBoxLayout()
        self._filter = QComboBox()
        for label, val in [("すべて", "all"), ("未確認のみ", "unreviewed"),
                           ("KEEPのみ", "keep"), ("REMOVEのみ", "remove")]:
            self._filter.addItem(label, val)
        self._filter.currentIndexChanged.connect(self._refresh_segment_table)
        self._sort = QComboBox()
        for label, val in [("大きい順", "area_desc"), ("小さい順", "area_asc"),
                           ("スコア順", "score")]:
            self._sort.addItem(label, val)
        self._sort.currentIndexChanged.connect(self._refresh_segment_table)
        filt.addWidget(self._filter)
        filt.addWidget(self._sort)
        rv.addLayout(filt)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(["ID", "面積", "IoU", "安定性", "判断"])
        self._table.itemSelectionChanged.connect(self._on_table_selection)
        rv.addWidget(self._table, 1)

        row = QHBoxLayout()
        self._btn_keep = QPushButton("必要(KEEP)")
        self._btn_remove = QPushButton("不要(REMOVE)")
        self._btn_unrev = QPushButton("未確認へ")
        self._btn_keep.clicked.connect(lambda: self._decide_current(SegmentDecision.KEEP))
        self._btn_remove.clicked.connect(lambda: self._decide_current(SegmentDecision.REMOVE))
        self._btn_unrev.clicked.connect(lambda: self._decide_current(SegmentDecision.UNREVIEWED))
        for b in (self._btn_keep, self._btn_remove, self._btn_unrev):
            row.addWidget(b)
        rv.addLayout(row)

        self._btn_complete = QPushButton("この画像のレビューを完了")
        self._btn_complete.clicked.connect(self._complete_review)
        rv.addWidget(self._btn_complete)

        gen = QHBoxLayout()
        self._final_mode = QComboBox()
        for label, val in _FINAL_MODE_ITEMS:
            self._final_mode.addItem(label, val)
        gen.addWidget(self._final_mode)
        self._btn_final = QPushButton("最終マスク生成")
        self._btn_final.clicked.connect(self._emit_final)
        gen.addWidget(self._btn_final)
        rv.addLayout(gen)

        self._btn_final_batch = QPushButton("レビュー済み画像へ一括生成")
        self._btn_final_batch.clicked.connect(self._emit_final_batch)
        rv.addWidget(self._btn_final_batch)

        splitter.addWidget(right)
        splitter.setSizes([220, 660, 320])

    # ------------------------------------------------------------------ #
    # 画像一覧
    # ------------------------------------------------------------------ #

    def _populate_image_list(self) -> None:
        self._image_list.clear()
        for item in self._items:
            status = item.get("status", "ready")
            icon = _STATUS_ICON.get(status, status)
            QListWidgetItem(f"{icon}  {item['image_key']}", self._image_list)

    def _update_list_row_label(self, row: int) -> None:
        item = self._items[row]
        status = item.get("status", "ready")
        if status == "ready" and item.get("review_completed"):
            status = "completed"
        elif status == "ready" and item.get("partial"):
            status = "partial"
        icon = _STATUS_ICON.get(status, status)
        self._image_list.item(row).setText(f"{icon}  {item['image_key']}")

    # ------------------------------------------------------------------ #
    # 画像切替・読込
    # ------------------------------------------------------------------ #

    def _on_image_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._items):
            return
        # 直前の画像の decisions を保存
        self._save_decisions()
        self._load_image(self._items[row])

    def _cache_dir(self, item: dict) -> Path:
        cid = item.get("cache_id") or amg_manifest.cache_id_for(item["image_key"])
        return (self._root / amg_manifest.CACHE_DIRNAME / amg_manifest.IMAGES_DIRNAME / cid)

    def _load_image(self, item: dict) -> None:
        self._cur_key = item["image_key"]
        cache_dir = self._cache_dir(item)
        npz_path = cache_dir / amg_manifest.SEGMENTS_NPZ_NAME
        self._cur_manifest_path = cache_dir / amg_manifest.MANIFEST_NAME
        try:
            self._npz = amg_npz.load_segments_npz(npz_path)
            manifest = amg_manifest.read_json(self._cur_manifest_path)
        except Exception as e:  # noqa: BLE001
            _log.error("レビュー画像読込失敗 %s: %s", self._cur_key, e)
            self._npz = None
            self._cache = None
            self._decisions = {}
            self._canvas.set_image(np.zeros((360, 480, 3), np.uint8))
            self._table.setRowCount(0)
            return

        seg_ids = np.asarray(self._npz["segment_ids"]).tolist()
        self._seg_index_by_id = {int(s): i for i, s in enumerate(seg_ids)}
        self._id_by_seg_index = {i: int(s) for i, s in enumerate(seg_ids)}
        self._decisions = normalize_decisions(
            manifest.get("review", {}).get("decisions", {}), segment_ids=seg_ids)
        self._cache = amg_hit_test.MaskDecodeCache(self._npz, max_size=self._decode_cache_size)
        self._current_seg_index = None
        self._candidates = []

        # 元画像
        src = manifest.get("source_path", item.get("source_path", ""))
        rgb = self._load_rgb(src, int(self._npz["image_shape"][1]), int(self._npz["image_shape"][0]))
        self._base_rgb = rgb
        self._canvas.set_image(rgb)

        self._refresh_segment_table()
        self._refresh_overlay()
        self._refresh_counts()

    @staticmethod
    def _load_rgb(src: str, w: int, h: int) -> np.ndarray:
        img = imread_jp(Path(src)) if src else None
        if img is None:
            return np.zeros((h, w, 3), np.uint8)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        # imread_jp は BGR -> RGB
        return np.ascontiguousarray(img[:, :, ::-1])

    # ------------------------------------------------------------------ #
    # 判断操作
    # ------------------------------------------------------------------ #

    def _on_canvas_clicked(self, x: int, y: int, button: int, ctrl: bool) -> None:
        if self._npz is None:
            return
        cands = amg_hit_test.candidates_at_point(self._npz, x, y)
        if not cands:
            return
        self._candidates = cands
        self._current_seg_index = cands[0]  # 最小候補を初期選択
        if ctrl:
            decision = SegmentDecision.UNREVIEWED
        elif button == 1:
            decision = SegmentDecision.KEEP
        else:
            decision = SegmentDecision.REMOVE
        self._apply_decision(self._current_seg_index, decision)

    def cycle_candidate(self, forward: bool = True) -> None:
        """Tab/Shift+Tab で重複候補を切替 (現在の選択を移動)。"""
        if not self._candidates:
            return
        self._current_seg_index = amg_hit_test.cycle_index(
            self._candidates, self._current_seg_index, forward=forward)
        self._refresh_overlay()
        self._select_table_row(self._current_seg_index)

    def keyPressEvent(self, e) -> None:
        if e.key() == Qt.Key.Key_Tab:
            self.cycle_candidate(forward=True)
            e.accept()
            return
        if e.key() == Qt.Key.Key_Backtab:
            self.cycle_candidate(forward=False)
            e.accept()
            return
        super().keyPressEvent(e)

    def _decide_current(self, decision: SegmentDecision) -> None:
        if self._current_seg_index is not None:
            self._apply_decision(self._current_seg_index, decision)

    def _apply_decision(self, seg_index: int, decision: SegmentDecision) -> None:
        sid = self._id_by_seg_index.get(seg_index)
        if sid is None:
            return
        self._decisions[str(sid)] = decision.value
        self._refresh_overlay()
        self._refresh_segment_table()
        self._refresh_counts()
        self._select_table_row(seg_index)

    def set_decision_by_id(self, segment_id: int, decision: SegmentDecision) -> None:
        """テスト/外部用: segment_id で判断を設定する。"""
        idx = self._seg_index_by_id.get(int(segment_id))
        if idx is not None:
            self._apply_decision(idx, decision)

    # ------------------------------------------------------------------ #
    # オーバーレイ
    # ------------------------------------------------------------------ #

    def _decode_union_indices(self, target: str) -> list[int]:
        return [i for i, sid in self._id_by_seg_index.items()
                if self._decisions.get(str(sid)) == target]

    def _refresh_overlay(self) -> None:
        if self._npz is None or self._cache is None:
            self._canvas.set_overlay(None)
            return
        h, w = self._cache.shape
        alpha = self._opacity.value() / 100.0
        overlay = np.zeros((h, w, 4), np.uint8)

        keep = self._cache.union(self._decode_union_indices(SegmentDecision.KEEP.value))
        remove = self._cache.union(self._decode_union_indices(SegmentDecision.REMOVE.value))
        self._paint(overlay, keep, _COL_KEEP, alpha)
        self._paint(overlay, remove, _COL_REMOVE, alpha)
        # 現在選択は明るい輪郭 (cyan, 高 alpha)
        if self._current_seg_index is not None:
            cur = self._cache.get(self._current_seg_index) > 0
            self._paint(overlay, cur, _COL_CURRENT, min(1.0, alpha + 0.25))
        self._canvas.set_overlay(overlay)

    @staticmethod
    def _paint(overlay: np.ndarray, mask: np.ndarray, color, alpha: float) -> None:
        if not mask.any():
            return
        a = int(max(0, min(255, alpha * 255)))
        overlay[mask, 0] = color[0]
        overlay[mask, 1] = color[1]
        overlay[mask, 2] = color[2]
        overlay[mask, 3] = a

    # ------------------------------------------------------------------ #
    # セグメント一覧
    # ------------------------------------------------------------------ #

    def _visible_indices(self) -> list[int]:
        if self._npz is None:
            return []
        n = int(np.asarray(self._npz["segment_ids"]).shape[0])
        flt = self._filter.currentData()
        idxs = list(range(n))
        if flt != "all":
            idxs = [i for i in idxs if self._decisions.get(str(self._id_by_seg_index[i])) == flt]
        area = np.asarray(self._npz["area"])
        iou = np.asarray(self._npz["predicted_iou"])
        stab = np.asarray(self._npz["stability_score"])
        sort = self._sort.currentData()
        if sort == "area_asc":
            idxs.sort(key=lambda i: int(area[i]))
        elif sort == "area_desc":
            idxs.sort(key=lambda i: -int(area[i]))
        else:  # score
            idxs.sort(key=lambda i: (-float(iou[i]), -float(stab[i])))
        return idxs

    def _refresh_segment_table(self) -> None:
        if self._npz is None:
            self._table.setRowCount(0)
            return
        area = np.asarray(self._npz["area"])
        iou = np.asarray(self._npz["predicted_iou"])
        stab = np.asarray(self._npz["stability_score"])
        idxs = self._visible_indices()
        self._table.blockSignals(True)
        self._table.setRowCount(len(idxs))
        self._row_to_seg_index = {}
        for row, i in enumerate(idxs):
            sid = self._id_by_seg_index[i]
            dec = self._decisions.get(str(sid), "unreviewed")
            cells = [str(sid), str(int(area[i])), f"{float(iou[i]):.3f}",
                     f"{float(stab[i]):.3f}", {"keep": "必要", "remove": "不要"}.get(dec, "未確認")]
            for col, text in enumerate(cells):
                self._table.setItem(row, col, QTableWidgetItem(text))
            self._row_to_seg_index[row] = i
        self._table.blockSignals(False)

    def _on_table_selection(self) -> None:
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return
        seg_index = getattr(self, "_row_to_seg_index", {}).get(rows[0].row())
        if seg_index is not None:
            self._current_seg_index = seg_index
            self._candidates = [seg_index]
            self._refresh_overlay()

    def _select_table_row(self, seg_index: int) -> None:
        mapping = getattr(self, "_row_to_seg_index", {})
        for row, i in mapping.items():
            if i == seg_index:
                self._table.blockSignals(True)
                self._table.selectRow(row)
                self._table.blockSignals(False)
                return

    def _refresh_counts(self) -> None:
        c = count_decisions(self._decisions)
        self._counts_label.setText(
            f"KEEP {c['keep']} / REMOVE {c['remove']} / 未確認 {c['unreviewed']}")

    # ------------------------------------------------------------------ #
    # 保存・完了・最終マスク
    # ------------------------------------------------------------------ #

    def _save_decisions(self, completed: Optional[bool] = None) -> None:
        if self._cur_manifest_path is None or self._npz is None:
            return
        try:
            amg_manifest.update_manifest_decisions(
                self._cur_manifest_path, self._decisions, completed=completed)
        except Exception as e:  # noqa: BLE001
            _log.error("decisions 保存失敗: %s", e)

    def _complete_review(self) -> None:
        if self._npz is None:
            return
        if not is_review_complete(self._decisions):
            c = count_decisions(self._decisions)
            ret = QMessageBox.question(
                self, "レビュー完了",
                f"未確認の候補が{c['unreviewed']}件あります。\n"
                "未確認候補を反映せず、レビューを完了しますか？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if ret != QMessageBox.StandardButton.Yes:
                return
        self._save_decisions(completed=True)
        row = self._image_list.currentRow()
        if 0 <= row < len(self._items):
            self._items[row]["review_completed"] = True
            self._update_list_row_label(row)

    def _emit_final(self) -> None:
        if self._cur_key is None:
            return
        self._save_decisions()
        self.final_mask_requested.emit(self._cur_key, self._final_mode.currentData())

    def _emit_final_batch(self) -> None:
        self._save_decisions()
        keys = [it["image_key"] for it in self._items if it.get("review_completed")]
        if not keys:
            QMessageBox.information(self, "一括生成", "レビュー完了済みの画像がありません。")
            return
        self.final_mask_batch_requested.emit(keys, self._final_mode.currentData())

    def closeEvent(self, e) -> None:
        self._save_decisions()
        super().closeEvent(e)
