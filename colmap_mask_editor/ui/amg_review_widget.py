"""
V0.8/V0.10 GUI: AMG 解析結果のレビュー画面 (QDialog)。

レビュー方式は 2 つ:
  - 不要領域だけ選択（推奨） = REMOVE_ONLY (V0.10)
      全画素を暗黙 KEEP とし、不要候補だけ REMOVE する。未確認は最終マスクに反映
      しない。最終マスク = 既存 amg_mask_composer.compose_final_mask(MODE_EXCLUDE_REMOVE)。
  - 必要・不要を個別設定（従来方式） = standard (V0.8)
      候補ごとに KEEP / REMOVE / 未確認 を設定する。

判断は manifest.json の review ブロックのみ原子更新 (NPZ は書き換えない)。重複候補の
グループ計算 (review_index) は GUI スレッド外 (AmgReviewIndexWorker) で行う。

torch / sam2 は import しない (numpy / cv2 / amg_* のみ)。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
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
from ai import amg_remove_only as ro
from ai import amg_review_index as ri
from ai.amg_review_history import ReviewAction, ReviewHistory, apply_undo
from ai.amg_review_state import (
    SegmentDecision,
    count_decisions,
    is_review_complete,
    normalize_decisions,
)
from core.amg_review_index_worker import AmgReviewIndexWorker
from core.mask_io import imread_jp

_log = logging.getLogger(__name__)

# 画像一覧の状態アイコン
_STATUS_ICON = {
    "unprocessed": "○ 未解析",
    "ready": "✓ 解析済み・未確認",
    "partial": "△ REMOVE設定あり・未完了",
    "completed": "● レビュー完了",
    "failed": "! 失敗",
    "stale": "↻ 古い",
    "corrupt": "✗ 破損",
}

# 候補色 (RGBA)
_COL_UNREVIEWED = (255, 230, 0)
_COL_KEEP = (0, 220, 60)
_COL_REMOVE = (235, 40, 40)
_COL_CURRENT = (0, 220, 255)
_COL_BASE_OUTSIDE = (40, 40, 40)

_FINAL_MODE_ITEMS = [
    ("不要領域を除外", amg_mask_composer.MODE_EXCLUDE_REMOVE),
    ("必要領域のみ", amg_mask_composer.MODE_KEEP_ONLY),
    ("現在マスクへ追加・除外", amg_mask_composer.MODE_ADD_REMOVE),
]

_WORKFLOW_ITEMS = [
    ("不要領域だけ選択（推奨）", ro.WORKFLOW_REMOVE_ONLY),
    ("必要・不要を個別設定（従来方式）", ro.WORKFLOW_STANDARD),
]

# REMOVE_ONLY の確認順 (意味分類ではない)
_RO_SORT_ITEMS = [
    ("大きい候補から", "area"),
    ("画像端に接する候補から", "edge"),
    ("品質スコア順", "quality"),
    ("確認順スコア", "priority"),
    ("元のSAM順", "sam"),
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
    final_mask_requested = Signal(str, str)        # image_key, mode (現在画像)
    final_mask_batch_requested = Signal(list, str)  # image_keys, mode
    index_ready = Signal(str)                      # image_key (review_index 構築完了)

    def __init__(self, project_root, image_items: list[dict], parent=None,
                 decode_cache_size: int = 12, *,
                 workflow: str = ro.WORKFLOW_REMOVE_ONLY,
                 base_mode: str = ro.BASE_EXISTING_OR_FULL,
                 iou_threshold: float = 0.85,
                 containment_threshold: float = 0.95,
                 covered_threshold: float = 0.98,
                 auto_advance: bool = True,
                 auto_next_image: bool = True,
                 representatives_only: bool = True,
                 hide_covered: bool = True,
                 default_sort: str = "priority",
                 undo_limit: int = 100,
                 show_base_outside: bool = True) -> None:
        super().__init__(parent)
        self.setWindowTitle("AMG レビュー — 不要領域だけ選択（推奨）")
        self.resize(1240, 800)
        self._root = Path(project_root)
        self._items = list(image_items)
        self._decode_cache_size = int(decode_cache_size)

        # 方式・設定
        self._workflow = workflow if workflow in ro.VALID_WORKFLOWS else ro.WORKFLOW_REMOVE_ONLY
        self._base_mode = base_mode if base_mode in ro.VALID_BASE_MODES else ro.BASE_EXISTING_OR_FULL
        self._iou_threshold = float(iou_threshold)
        self._containment_threshold = float(containment_threshold)
        self._covered_threshold = float(covered_threshold)
        self._opt_auto_advance = bool(auto_advance)
        self._opt_auto_next_image = bool(auto_next_image)
        self._opt_representatives_only = bool(representatives_only)
        self._opt_hide_covered = bool(hide_covered)
        self._default_sort = default_sort
        self._show_base_outside_default = bool(show_base_outside)

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
        self._base_mask: Optional[np.ndarray] = None    # bool (h,w) 基準マスク
        self._remove_union: Optional[np.ndarray] = None  # bool (h,w) REMOVE 和集合
        self._row_to_seg_index: dict[int, int] = {}
        self._visible_order: list[int] = []
        self._show_current = True

        # review_index (グループ・確認順)
        self._index: Optional[dict[str, np.ndarray]] = None
        self._rep_indices: set[int] = set()
        self._index_worker: Optional[AmgReviewIndexWorker] = None
        self._index_total_candidates = 0
        self._index_group_count = 0

        # 判断 Undo (通常マスクの Undo とは分離)
        self._history = ReviewHistory(limit=int(undo_limit))

        self._build_ui()
        self._apply_workflow_ui()
        self._populate_image_list()
        if self._items:
            self._image_list.setCurrentRow(0)

    # ------------------------------------------------------------------ #
    # UI 構築
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        # レビュー方式セレクタ (上部)
        top = QHBoxLayout()
        top.addWidget(QLabel("レビュー方式:"))
        self._workflow_combo = QComboBox()
        for label, val in _WORKFLOW_ITEMS:
            self._workflow_combo.addItem(label, val)
        idx = next((i for i in range(self._workflow_combo.count())
                    if self._workflow_combo.itemData(i) == self._workflow), 0)
        self._workflow_combo.setCurrentIndex(idx)
        self._workflow_combo.currentIndexChanged.connect(self._on_workflow_changed)
        top.addWidget(self._workflow_combo)
        top.addStretch(1)
        root.addLayout(top)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)

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

        # 表示切替チェック
        toggles = QHBoxLayout()
        self._chk_show_remove = QCheckBox("REMOVE済み領域")
        self._chk_show_remove.setChecked(True)
        self._chk_show_current = QCheckBox("現在候補")
        self._chk_show_current.setChecked(True)
        self._chk_show_base_outside = QCheckBox("基準マスク外")
        self._chk_show_base_outside.setChecked(self._show_base_outside_default)
        for c in (self._chk_show_remove, self._chk_show_current, self._chk_show_base_outside):
            c.stateChanged.connect(lambda _v: self._refresh_overlay())
            toggles.addWidget(c)
        toggles.addStretch(1)
        cv.addLayout(toggles)

        ctl = QHBoxLayout()
        ctl.addWidget(QLabel("透明度"))
        self._opacity = QSlider(Qt.Orientation.Horizontal)
        self._opacity.setRange(0, 100)
        self._opacity.setValue(50)
        self._opacity.valueChanged.connect(lambda _v: self._refresh_overlay())
        ctl.addWidget(self._opacity)
        self._hint = QLabel("")
        self._hint.setStyleSheet("color:#9cf;")
        ctl.addWidget(self._hint, 1)
        cv.addLayout(ctl)
        splitter.addWidget(center)

        # 右: 進捗 / フィルタ / 候補一覧 / 操作
        right = QWidget()
        rv = QVBoxLayout(right)
        self._counts_label = QLabel("")
        self._counts_label.setWordWrap(True)
        rv.addWidget(self._counts_label)

        # REMOVE_ONLY 用フィルタ
        self._chk_reps_only = QCheckBox("代表候補だけ表示")
        self._chk_reps_only.setChecked(self._opt_representatives_only)
        self._chk_reps_only.stateChanged.connect(lambda _v: self._refresh_segment_table())
        self._chk_hide_covered = QCheckBox("すでに除外済みの候補を隠す")
        self._chk_hide_covered.setChecked(self._opt_hide_covered)
        self._chk_hide_covered.stateChanged.connect(lambda _v: self._refresh_segment_table())
        self._chk_auto_advance = QCheckBox("判断後に次候補へ移動")
        self._chk_auto_advance.setChecked(self._opt_auto_advance)
        self._chk_auto_next_image = QCheckBox("完了後に次の未完了画像へ移動")
        self._chk_auto_next_image.setChecked(self._opt_auto_next_image)
        rv.addWidget(self._chk_reps_only)
        rv.addWidget(self._chk_hide_covered)
        rv.addWidget(self._chk_auto_advance)
        rv.addWidget(self._chk_auto_next_image)

        filt = QHBoxLayout()
        self._filter = QComboBox()
        for label, val in [("すべて", "all"), ("未確認のみ", "unreviewed"),
                           ("KEEPのみ", "keep"), ("REMOVEのみ", "remove")]:
            self._filter.addItem(label, val)
        self._filter.currentIndexChanged.connect(self._refresh_segment_table)
        self._sort = QComboBox()
        self._sort.currentIndexChanged.connect(self._refresh_segment_table)
        filt.addWidget(self._filter)
        filt.addWidget(self._sort)
        rv.addLayout(filt)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(["ID", "面積", "IoU", "安定性", "判断"])
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.itemSelectionChanged.connect(self._on_table_selection)
        rv.addWidget(self._table, 1)

        # 判断ボタン
        row = QHBoxLayout()
        self._btn_remove = QPushButton("不要として除外")
        self._btn_unremove = QPushButton("除外を解除")
        self._btn_keep = QPushButton("必要(KEEP)")
        self._btn_unrev = QPushButton("未確認へ")
        self._btn_remove.clicked.connect(self._on_btn_remove)
        self._btn_unremove.clicked.connect(self._on_btn_unremove)
        self._btn_keep.clicked.connect(lambda: self._decide_current(SegmentDecision.KEEP))
        self._btn_unrev.clicked.connect(lambda: self._decide_current(SegmentDecision.UNREVIEWED))
        for b in (self._btn_remove, self._btn_unremove, self._btn_keep, self._btn_unrev):
            row.addWidget(b)
        rv.addLayout(row)

        nav = QHBoxLayout()
        self._btn_next = QPushButton("次の候補")
        self._btn_prev = QPushButton("前の候補")
        self._btn_undo = QPushButton("判断を元に戻す")
        self._btn_next.clicked.connect(lambda: self._goto_relative(forward=True))
        self._btn_prev.clicked.connect(lambda: self._goto_relative(forward=False))
        self._btn_undo.clicked.connect(self._undo_decision)
        for b in (self._btn_prev, self._btn_next, self._btn_undo):
            nav.addWidget(b)
        rv.addLayout(nav)

        bulk = QHBoxLayout()
        self._btn_bulk_remove = QPushButton("選択をまとめてREMOVE")
        self._btn_bulk_unremove = QPushButton("選択のREMOVEを解除")
        self._btn_clear_all = QPushButton("この画像の全REMOVE解除")
        self._btn_bulk_remove.clicked.connect(self._bulk_remove_selected)
        self._btn_bulk_unremove.clicked.connect(self._bulk_unremove_selected)
        self._btn_clear_all.clicked.connect(self._clear_all_remove)
        for b in (self._btn_bulk_remove, self._btn_bulk_unremove, self._btn_clear_all):
            bulk.addWidget(b)
        rv.addLayout(bulk)

        self._btn_complete = QPushButton("レビュー完了")
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
        splitter.setSizes([220, 660, 360])

    def _apply_workflow_ui(self) -> None:
        """方式に応じて UI の表示・選択肢を切り替える。"""
        remove_only = self._workflow == ro.WORKFLOW_REMOVE_ONLY
        # REMOVE_ONLY では KEEP / 未確認ボタンと従来フィルタを隠す
        self._btn_keep.setVisible(not remove_only)
        self._btn_unrev.setVisible(not remove_only)
        self._filter.setVisible(not remove_only)
        self._chk_reps_only.setVisible(remove_only)
        self._chk_hide_covered.setVisible(remove_only)
        self._chk_auto_advance.setVisible(remove_only)
        self._chk_auto_next_image.setVisible(remove_only)
        self._btn_remove.setVisible(remove_only)
        self._btn_unremove.setVisible(remove_only)
        self._btn_bulk_remove.setVisible(remove_only)
        self._btn_bulk_unremove.setVisible(remove_only)
        self._btn_clear_all.setVisible(remove_only)
        self._chk_show_remove.setVisible(remove_only)
        self._chk_show_base_outside.setVisible(remove_only)

        # ソート選択肢
        self._sort.blockSignals(True)
        self._sort.clear()
        if remove_only:
            for label, val in _RO_SORT_ITEMS:
                self._sort.addItem(label, val)
            sidx = next((i for i in range(self._sort.count())
                         if self._sort.itemData(i) == self._default_sort), 0)
            self._sort.setCurrentIndex(sidx)
            self._table.setHorizontalHeaderLabels(["ID", "面積", "品質", "端", "判断"])
            self._hint.setText("左=不要 / 右=解除 / Ctrl+左=解除 / R=除外 U=解除 "
                               "Enter=除外して次 N/P=候補移動 Space=現在候補表示")
            # REMOVE_ONLY は最終マスク方式を「不要領域を除外」固定
            fidx = next((i for i in range(self._final_mode.count())
                         if self._final_mode.itemData(i) == amg_mask_composer.MODE_EXCLUDE_REMOVE), 0)
            self._final_mode.setCurrentIndex(fidx)
            self._final_mode.setEnabled(False)
            self._btn_complete.setText("レビュー完了")
        else:
            for label, val in [("大きい順", "area_desc"), ("小さい順", "area_asc"),
                               ("スコア順", "score")]:
                self._sort.addItem(label, val)
            self._table.setHorizontalHeaderLabels(["ID", "面積", "IoU", "安定性", "判断"])
            self._hint.setText("左=必要 / 右=不要 / Ctrl+左=未確認 / Tab=重複候補切替")
            self._final_mode.setEnabled(True)
            self._btn_complete.setText("この画像のレビューを完了")
        self._sort.blockSignals(False)

    def _on_workflow_changed(self, _idx: int) -> None:
        self._save_decisions()
        self._workflow = self._workflow_combo.currentData()
        self._apply_workflow_ui()
        self._refresh_segment_table()
        self._refresh_overlay()
        self._refresh_counts()

    # ------------------------------------------------------------------ #
    # 画像一覧
    # ------------------------------------------------------------------ #

    def _populate_image_list(self) -> None:
        self._image_list.clear()
        for item in self._items:
            QListWidgetItem(self._list_label(item), self._image_list)

    def _list_label(self, item: dict) -> str:
        status = item.get("status", "ready")
        if status == "ready" and item.get("review_completed"):
            status = "completed"
        elif status == "ready" and item.get("partial"):
            status = "partial"
        icon = _STATUS_ICON.get(status, status)
        return f"{icon}  {item['image_key']}"

    def _update_list_row_label(self, row: int) -> None:
        if 0 <= row < len(self._items):
            self._image_list.item(row).setText(self._list_label(self._items[row]))

    # ------------------------------------------------------------------ #
    # 画像切替・読込
    # ------------------------------------------------------------------ #

    def _on_image_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._items):
            return
        self._save_decisions()
        self._cancel_index_worker()
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
        self._history.clear()
        self._index = None
        self._rep_indices = set()
        self._index_total_candidates = len(seg_ids)
        self._index_group_count = 0
        self._show_current = True

        # 元画像
        src = manifest.get("source_path", item.get("source_path", ""))
        rgb = self._load_rgb(src, int(self._npz["image_shape"][1]), int(self._npz["image_shape"][0]))
        self._base_rgb = rgb
        self._canvas.set_image(rgb)

        # 基準マスク (REMOVE_ONLY)
        self._base_mask = self._resolve_base_mask(item)
        self._remove_union = self._compute_remove_union()

        self._refresh_segment_table()
        self._refresh_overlay()
        self._refresh_counts()

        # 重複グループ計算は GUI スレッド外で
        self._start_index_worker(cache_dir)

    @staticmethod
    def _load_rgb(src: str, w: int, h: int) -> np.ndarray:
        img = imread_jp(Path(src)) if src else None
        if img is None:
            return np.zeros((h, w, 3), np.uint8)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        return np.ascontiguousarray(img[:, :, ::-1])

    def _resolve_base_mask(self, item: dict) -> Optional[np.ndarray]:
        if self._npz is None:
            return None
        h, w = int(self._npz["image_shape"][0]), int(self._npz["image_shape"][1])
        existing = None
        mask_path = item.get("mask_path")
        if self._base_mode == ro.BASE_EXISTING_OR_FULL and mask_path and Path(mask_path).exists():
            img = imread_jp(Path(mask_path))
            if img is not None:
                if img.ndim >= 3:
                    img = img[:, :, 0]
                if img.shape == (h, w):
                    existing = img
                else:
                    _log.warning("既存マスクのサイズ不一致のため基準を全面にします: %s", item.get("image_key"))
        try:
            return ro.resolve_base_mask(
                h, w, existing,
                ro.BASE_EXISTING_OR_FULL if existing is not None else ro.BASE_FULL)
        except ro.BaseMaskSizeMismatch:
            return np.ones((h, w), dtype=bool)

    # ------------------------------------------------------------------ #
    # review_index Worker
    # ------------------------------------------------------------------ #

    def _start_index_worker(self, cache_dir: Path) -> None:
        if AmgReviewIndexWorker is None:
            return
        worker = AmgReviewIndexWorker(
            str(cache_dir), cache_id=self._cur_key or "",
            iou_threshold=self._iou_threshold,
            containment_threshold=self._containment_threshold, parent=self)
        worker.finished_ok.connect(self._on_index_ready)
        worker.failed.connect(lambda cid, msg: _log.warning("review_index 失敗 %s: %s", cid, msg))
        self._index_worker = worker
        worker.start()

    def _cancel_index_worker(self) -> None:
        if self._index_worker is not None:
            self._index_worker.cancel()
            self._index_worker.wait(2000)
            self._index_worker = None

    def _on_index_ready(self, cache_id: str, info: dict) -> None:
        if cache_id != self._cur_key or self._npz is None:
            return
        try:
            item = next(it for it in self._items if it["image_key"] == cache_id)
            cache_dir = self._cache_dir(item)
            self._index = ri.load_review_index(cache_dir / ri.REVIEW_INDEX_NPZ_NAME)
        except Exception as e:  # noqa: BLE001
            _log.warning("review_index 読込失敗: %s", e)
            return
        seg_ids = np.asarray(self._index["segment_ids"])
        reps = np.asarray(self._index["representative_segment_ids"])
        self._rep_indices = {i for i in range(seg_ids.size)
                             if int(reps[i]) == int(seg_ids[i])}
        self._index_group_count = int(info.get("group_count", len(self._rep_indices)))
        self._index_total_candidates = int(info.get("segment_count", seg_ids.size))
        self._refresh_segment_table()
        self._refresh_counts()
        self.index_ready.emit(cache_id)

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
        self._current_seg_index = cands[0]
        if self._workflow == ro.WORKFLOW_REMOVE_ONLY:
            if ctrl or button == 2:
                self._set_decision_recorded(self._current_seg_index, SegmentDecision.UNREVIEWED)
            else:
                self._set_decision_recorded(self._current_seg_index, SegmentDecision.REMOVE)
                self._maybe_auto_advance()
        else:
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
        key = e.key()
        if key == Qt.Key.Key_Tab:
            self.cycle_candidate(forward=True)
            e.accept(); return
        if key == Qt.Key.Key_Backtab:
            self.cycle_candidate(forward=False)
            e.accept(); return
        if self._workflow == ro.WORKFLOW_REMOVE_ONLY:
            if key == Qt.Key.Key_R:
                self._on_btn_remove(); e.accept(); return
            if key == Qt.Key.Key_U:
                self._on_btn_unremove(); e.accept(); return
            if key == Qt.Key.Key_N:
                self._goto_relative(forward=True); e.accept(); return
            if key == Qt.Key.Key_P:
                self._goto_relative(forward=False); e.accept(); return
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if self._current_seg_index is not None:
                    self._set_decision_recorded(self._current_seg_index, SegmentDecision.REMOVE)
                    self._goto_relative(forward=True)
                e.accept(); return
            if key == Qt.Key.Key_Space:
                self._show_current = not self._show_current
                self._refresh_overlay(); e.accept(); return
        super().keyPressEvent(e)

    # ----- ボタン -----

    def _on_btn_remove(self) -> None:
        if self._current_seg_index is not None:
            self._set_decision_recorded(self._current_seg_index, SegmentDecision.REMOVE)
            self._maybe_auto_advance()

    def _on_btn_unremove(self) -> None:
        if self._current_seg_index is not None:
            self._set_decision_recorded(self._current_seg_index, SegmentDecision.UNREVIEWED)
            self._maybe_auto_advance()

    def _decide_current(self, decision: SegmentDecision) -> None:
        if self._current_seg_index is not None:
            if self._workflow == ro.WORKFLOW_REMOVE_ONLY:
                self._set_decision_recorded(self._current_seg_index, decision)
            else:
                self._apply_decision(self._current_seg_index, decision)

    def _maybe_auto_advance(self) -> None:
        if self._workflow == ro.WORKFLOW_REMOVE_ONLY and self._chk_auto_advance.isChecked():
            self._goto_next_unremoved()

    # ----- 判断適用 -----

    def _set_decision_recorded(self, seg_index: int, decision: SegmentDecision) -> None:
        """REMOVE_ONLY: 判断を Undo 履歴へ記録して適用する。"""
        sid = self._id_by_seg_index.get(seg_index)
        if sid is None:
            return
        before = self._decisions.get(str(sid), SegmentDecision.UNREVIEWED.value)
        after = decision.value
        self._history.record(ReviewAction(int(sid), before, after))
        self._decisions[str(sid)] = after
        self._after_decisions_changed(seg_index)

    def _apply_decision(self, seg_index: int, decision: SegmentDecision) -> None:
        """standard: 履歴なしで適用 (従来動作)。"""
        sid = self._id_by_seg_index.get(seg_index)
        if sid is None:
            return
        self._decisions[str(sid)] = decision.value
        self._refresh_overlay()
        self._refresh_segment_table()
        self._refresh_counts()
        self._select_table_row(seg_index)

    def _after_decisions_changed(self, focus_index: Optional[int] = None) -> None:
        self._remove_union = self._compute_remove_union()
        self._refresh_overlay()
        self._refresh_segment_table()
        self._refresh_counts()
        if focus_index is not None:
            self._select_table_row(focus_index)

    def set_decision_by_id(self, segment_id: int, decision: SegmentDecision) -> None:
        """テスト/外部用: segment_id で判断を設定する。"""
        idx = self._seg_index_by_id.get(int(segment_id))
        if idx is None:
            return
        if self._workflow == ro.WORKFLOW_REMOVE_ONLY:
            self._set_decision_recorded(idx, decision)
        else:
            self._apply_decision(idx, decision)

    # ----- Undo -----

    def _undo_decision(self) -> None:
        if not self._history.can_undo():
            return
        step = self._history.undo()
        self._decisions = normalize_decisions(
            apply_undo(self._decisions, step),
            segment_ids=list(self._id_by_seg_index.values()))
        self._after_decisions_changed()

    # ----- 一括 -----

    def _selected_seg_indices(self) -> list[int]:
        rows = self._table.selectionModel().selectedRows()
        out = []
        for r in rows:
            i = self._row_to_seg_index.get(r.row())
            if i is not None:
                out.append(i)
        return out

    def _bulk_apply(self, seg_indices: list[int], decision: SegmentDecision) -> None:
        actions = []
        for i in seg_indices:
            sid = self._id_by_seg_index.get(i)
            if sid is None:
                continue
            before = self._decisions.get(str(sid), SegmentDecision.UNREVIEWED.value)
            actions.append(ReviewAction(int(sid), before, decision.value))
            self._decisions[str(sid)] = decision.value
        if actions:
            self._history.record(actions)   # 一括は 1 ステップ
            self._after_decisions_changed()

    def _bulk_remove_selected(self) -> None:
        self._bulk_apply(self._selected_seg_indices(), SegmentDecision.REMOVE)

    def _bulk_unremove_selected(self) -> None:
        self._bulk_apply(self._selected_seg_indices(), SegmentDecision.UNREVIEWED)

    def _clear_all_remove(self) -> None:
        idxs = [self._seg_index_by_id[sid] for sid in ro.remove_segment_ids(self._decisions)
                if sid in self._seg_index_by_id]
        self._bulk_apply(idxs, SegmentDecision.UNREVIEWED)

    # ----- 候補移動 -----

    def _goto_relative(self, forward: bool) -> None:
        if not self._visible_order:
            return
        if self._current_seg_index in self._visible_order:
            pos = self._visible_order.index(self._current_seg_index)
            pos = (pos + 1) % len(self._visible_order) if forward else (pos - 1) % len(self._visible_order)
        else:
            pos = 0 if forward else len(self._visible_order) - 1
        self._current_seg_index = self._visible_order[pos]
        self._candidates = [self._current_seg_index]
        self._refresh_overlay()
        self._select_table_row(self._current_seg_index)

    def _goto_next_unremoved(self) -> None:
        if not self._visible_order:
            return
        start = 0
        if self._current_seg_index in self._visible_order:
            start = self._visible_order.index(self._current_seg_index) + 1
        n = len(self._visible_order)
        for off in range(n):
            i = self._visible_order[(start + off) % n]
            sid = self._id_by_seg_index.get(i)
            if self._decisions.get(str(sid)) != SegmentDecision.REMOVE.value:
                self._current_seg_index = i
                self._candidates = [i]
                self._refresh_overlay()
                self._select_table_row(i)
                return

    # ------------------------------------------------------------------ #
    # オーバーレイ
    # ------------------------------------------------------------------ #

    def _compute_remove_union(self) -> Optional[np.ndarray]:
        if self._cache is None:
            return None
        idxs = [i for i, sid in self._id_by_seg_index.items()
                if self._decisions.get(str(sid)) == SegmentDecision.REMOVE.value]
        return self._cache.union(idxs)

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

        if self._workflow == ro.WORKFLOW_REMOVE_ONLY:
            if self._chk_show_base_outside.isChecked() and self._base_mask is not None:
                self._paint(overlay, ~self._base_mask, _COL_BASE_OUTSIDE, min(1.0, alpha + 0.2))
            if self._chk_show_remove.isChecked():
                rem = self._remove_union if self._remove_union is not None else self._compute_remove_union()
                if rem is not None:
                    self._paint(overlay, rem, _COL_REMOVE, alpha)
            if self._chk_show_current.isChecked() and self._show_current \
                    and self._current_seg_index is not None:
                cur = self._cache.get(self._current_seg_index) > 0
                self._paint(overlay, cur, _COL_CURRENT, min(1.0, alpha + 0.25))
        else:
            keep = self._cache.union(self._decode_union_indices(SegmentDecision.KEEP.value))
            remove = self._cache.union(self._decode_union_indices(SegmentDecision.REMOVE.value))
            self._paint(overlay, keep, _COL_KEEP, alpha)
            self._paint(overlay, remove, _COL_REMOVE, alpha)
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
    # 候補一覧
    # ------------------------------------------------------------------ #

    def _visible_indices(self) -> list[int]:
        if self._npz is None:
            return []
        n = int(np.asarray(self._npz["segment_ids"]).shape[0])
        idxs = list(range(n))
        area = np.asarray(self._npz["area"])

        if self._workflow == ro.WORKFLOW_REMOVE_ONLY:
            # 代表候補のみ / covered 抑制
            if self._chk_reps_only.isChecked() and self._rep_indices:
                idxs = [i for i in idxs if i in self._rep_indices]
            if self._chk_hide_covered.isChecked() and self._remove_union is not None \
                    and self._remove_union.any() and self._cache is not None:
                kept = []
                for i in idxs:
                    sid = self._id_by_seg_index.get(i)
                    if self._decisions.get(str(sid)) == SegmentDecision.REMOVE.value:
                        kept.append(i)   # REMOVE 済みは隠さない
                        continue
                    seg = self._cache.get(i) > 0
                    if not ro.is_covered(seg, self._remove_union, self._covered_threshold):
                        kept.append(i)
                idxs = kept
            idxs = self._sort_remove_only(idxs)
        else:
            flt = self._filter.currentData()
            if flt and flt != "all":
                idxs = [i for i in idxs if self._decisions.get(str(self._id_by_seg_index[i])) == flt]
            sort = self._sort.currentData()
            iou = np.asarray(self._npz["predicted_iou"])
            stab = np.asarray(self._npz["stability_score"])
            if sort == "area_asc":
                idxs.sort(key=lambda i: int(area[i]))
            elif sort == "area_desc":
                idxs.sort(key=lambda i: -int(area[i]))
            else:
                idxs.sort(key=lambda i: (-float(iou[i]), -float(stab[i])))
        return idxs

    def _sort_remove_only(self, idxs: list[int]) -> list[int]:
        area = np.asarray(self._npz["area"])
        mode = self._sort.currentData() or self._default_sort
        if self._index is not None:
            pri = np.asarray(self._index["priority_scores"])
            qual = np.asarray(self._index["quality_scores"])
            edge = np.asarray(self._index["edge_touch_flags"])
        else:
            pri = qual = edge = None
        if mode == "area" or pri is None:
            idxs.sort(key=lambda i: -int(area[i]))
        elif mode == "edge":
            idxs.sort(key=lambda i: (-int(edge[i]), -float(pri[i])))
        elif mode == "quality":
            idxs.sort(key=lambda i: -float(qual[i]))
        elif mode == "sam":
            idxs.sort(key=lambda i: self._id_by_seg_index.get(i, i))
        else:  # priority
            idxs.sort(key=lambda i: -float(pri[i]))
        return idxs

    def _refresh_segment_table(self) -> None:
        if self._npz is None:
            self._table.setRowCount(0)
            self._visible_order = []
            return
        area = np.asarray(self._npz["area"])
        idxs = self._visible_indices()
        self._visible_order = list(idxs)
        remove_only = self._workflow == ro.WORKFLOW_REMOVE_ONLY
        if remove_only and self._index is not None:
            qual = np.asarray(self._index["quality_scores"])
            edge = np.asarray(self._index["edge_touch_flags"])
        else:
            qual = edge = None
        iou = np.asarray(self._npz["predicted_iou"])
        stab = np.asarray(self._npz["stability_score"])

        self._table.blockSignals(True)
        self._table.setRowCount(len(idxs))
        self._row_to_seg_index = {}
        for row, i in enumerate(idxs):
            sid = self._id_by_seg_index[i]
            dec = self._decisions.get(str(sid), "unreviewed")
            if remove_only:
                col3 = f"{float(qual[i]):.3f}" if qual is not None else f"{float(iou[i]):.3f}"
                col4 = ("●" if (edge is not None and edge[i]) else "")
                label = {"remove": "不要"}.get(dec, "未確認")
            else:
                col3 = f"{float(iou[i]):.3f}"
                col4 = f"{float(stab[i]):.3f}"
                label = {"keep": "必要", "remove": "不要"}.get(dec, "未確認")
            cells = [str(sid), str(int(area[i])), col3, col4, label]
            for col, text in enumerate(cells):
                self._table.setItem(row, col, QTableWidgetItem(text))
            self._row_to_seg_index[row] = i
        self._table.blockSignals(False)

    def _on_table_selection(self) -> None:
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return
        seg_index = self._row_to_seg_index.get(rows[-1].row())
        if seg_index is not None:
            self._current_seg_index = seg_index
            self._candidates = [seg_index]
            self._refresh_overlay()

    def _select_table_row(self, seg_index: int) -> None:
        for row, i in self._row_to_seg_index.items():
            if i == seg_index:
                self._table.blockSignals(True)
                self._table.selectRow(row)
                self._table.blockSignals(False)
                return

    def _refresh_counts(self) -> None:
        if self._npz is None:
            self._counts_label.setText("")
            return
        if self._workflow == ro.WORKFLOW_REMOVE_ONLY:
            remove_n = ro.count_remove(self._decisions)
            shape = self._cache.shape if self._cache is not None else (0, 0)
            base = self._base_mask if self._base_mask is not None else np.ones(shape, bool)
            rem = self._remove_union if self._remove_union is not None else np.zeros(shape, bool)
            st = ro.pixel_stats(base, rem)
            total_cands = self._index_total_candidates
            review_cands = len(self._rep_indices) if self._rep_indices else total_cands
            self._counts_label.setText(
                f"REMOVE指定: {remove_n}候補\n"
                f"除外画素: {st.excluded_px:,} px  ({st.excluded_ratio * 100:.1f}%)\n"
                f"有効画素: {st.effective_px:,} px  ({st.effective_ratio * 100:.1f}%)\n"
                f"候補総数: {total_cands}   確認対象候補: {review_cands}")
        else:
            c = count_decisions(self._decisions)
            self._counts_label.setText(
                f"KEEP {c['keep']} / REMOVE {c['remove']} / 未確認 {c['unreviewed']}")

    # ------------------------------------------------------------------ #
    # 保存・完了・最終マスク
    # ------------------------------------------------------------------ #

    def _ui_state(self) -> dict:
        return {
            "representatives_only": self._chk_reps_only.isChecked(),
            "hide_covered": self._chk_hide_covered.isChecked(),
            "auto_advance": self._chk_auto_advance.isChecked(),
            "sort_mode": self._sort.currentData(),
        }

    def _save_decisions(self, completed: Optional[bool] = None) -> None:
        if self._cur_manifest_path is None or self._npz is None:
            return
        # 有効 segment_id 一覧は不変の NPZ から取得する。保存で最小化された
        # 既存 decisions のキーに依存しない (追加 REMOVE が消えないように)。
        valid_ids = self._npz["segment_ids"].tolist()
        try:
            if self._workflow == ro.WORKFLOW_REMOVE_ONLY:
                amg_manifest.update_manifest_review(
                    self._cur_manifest_path,
                    decisions=ro.prune_remove_only_decisions(self._decisions),
                    workflow=ro.WORKFLOW_REMOVE_ONLY,
                    base_mode=self._base_mode,
                    ui=self._ui_state(),
                    completed=completed,
                    valid_segment_ids=valid_ids)
            else:
                amg_manifest.update_manifest_decisions(
                    self._cur_manifest_path, self._decisions, completed=completed,
                    valid_segment_ids=valid_ids)
        except Exception as e:  # noqa: BLE001
            _log.error("decisions 保存失敗: %s", e)

    def _complete_review(self) -> None:
        if self._npz is None:
            return
        if self._workflow == ro.WORKFLOW_REMOVE_ONLY:
            remove_n = ro.count_remove(self._decisions)
            untouched = self._index_total_candidates - remove_n
            shape = self._cache.shape if self._cache is not None else (0, 0)
            base = self._base_mask if self._base_mask is not None else np.ones(shape, bool)
            rem = self._remove_union if self._remove_union is not None else np.zeros(shape, bool)
            st = ro.pixel_stats(base, rem)
            ret = QMessageBox.question(
                self, "レビュー完了",
                f"REMOVE候補: {remove_n}件\n"
                f"除外率: {st.excluded_ratio * 100:.1f}%\n"
                f"未操作候補: {untouched}件\n\n"
                "未操作候補はKEEPとして扱われます。\nレビューを完了しますか？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if ret != QMessageBox.StandardButton.Yes:
                return
        else:
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
        self._maybe_advance_image(row)

    def _maybe_advance_image(self, current_row: int) -> None:
        if self._workflow != ro.WORKFLOW_REMOVE_ONLY:
            return
        if not self._chk_auto_next_image.isChecked():
            return
        for r in range(current_row + 1, len(self._items)):
            if not self._items[r].get("review_completed"):
                self._image_list.setCurrentRow(r)
                return

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
        self._cancel_index_worker()
        super().closeEvent(e)
