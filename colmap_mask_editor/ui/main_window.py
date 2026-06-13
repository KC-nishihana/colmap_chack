"""
メインウィンドウ: 全パネルの配置・操作統括・ショートカット・保存・ログ
v0.4A.1: GrabCut Workerスレッド管理・プログレス表示・UI制御・大画像設定追加
"""

import csv
import datetime
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import QThread, Qt
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from core.mask_io import (
    get_colmap_mask_path,
    get_edited_mask_path,
    get_source_mask_save_path,
    imread_jp,
    load_mask_or_empty,
    save_mask,
)
from core.mask_ops import MaskEditor
from core.project_loader import ImageEntry, ProjectInfo, load_project
from ui.image_canvas import EditMode, ImageCanvas
from ui.image_list_panel import ImageListPanel

_log = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """アプリケーションのメインウィンドウ"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("COLMAP Mask Editor v0.4A.1")
        self.resize(1440, 900)

        self._project: Optional[ProjectInfo] = None
        self._current_index: int = -1
        self._editor: Optional[MaskEditor] = None
        self._save_colmap: bool = False

        # GrabCut Workerスレッド管理
        self._grabcut_thread: Optional[QThread] = None
        self._grabcut_worker = None           # GrabCutWorker (型循環回避)
        self._grabcut_request_id: int = 0    # リクエストID (インクリメント)
        self._grabcut_pending_mode: str = "add"
        self._grabcut_progress_dlg: Optional[QProgressDialog] = None

        self._setup_menu()
        self._setup_central()
        self._setup_shortcuts()

        self.statusBar().showMessage("プロジェクトフォルダを開いてください  [File > Open Project]")

    # ------------------------------------------------------------------ #
    # UI構築
    # ------------------------------------------------------------------ #

    def _setup_menu(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("ファイル(&F)")

        self._act_open = QAction("プロジェクトを開く(&O)...", self)
        self._act_open.setShortcut(QKeySequence("Ctrl+O"))
        self._act_open.triggered.connect(self._open_project)
        file_menu.addAction(self._act_open)

        self._act_save = QAction("保存(&S)", self)
        self._act_save.setShortcut(QKeySequence("Ctrl+S"))
        self._act_save.triggered.connect(self._save_current)
        file_menu.addAction(self._act_save)

        self._act_save_all = QAction("すべて保存(&A)", self)
        self._act_save_all.triggered.connect(self._save_all)
        file_menu.addAction(self._act_save_all)

        file_menu.addSeparator()

        exit_act = QAction("終了(&Q)", self)
        exit_act.setShortcut(QKeySequence("Ctrl+Q"))
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)

    def _setup_central(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左: 画像一覧
        self._list_panel = ImageListPanel()
        self._list_panel.setMinimumWidth(200)
        self._list_panel.setMaximumWidth(340)
        self._list_panel.image_selected.connect(self._on_image_selected)
        splitter.addWidget(self._list_panel)

        # 中央: キャンバス
        self._canvas = ImageCanvas()
        self._canvas.mask_changed.connect(self._on_mask_changed)
        self._canvas.mode_changed.connect(self._on_mode_changed)
        self._canvas.status_message.connect(self._on_canvas_status_message)
        self._canvas.grabcut_requested.connect(self._on_grabcut_requested)
        self._canvas.grabcut_cancel_requested.connect(self._cancel_grabcut)
        splitter.addWidget(self._canvas)

        # 右: コントロールパネル
        right_panel = self._build_right_panel()
        scroll = QScrollArea()
        scroll.setWidget(right_panel)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(210)
        scroll.setMaximumWidth(310)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        splitter.addWidget(scroll)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)

        self.setCentralWidget(splitter)

    def _build_right_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # ----- 編集モード -----
        mode_group = QGroupBox("編集モード")
        mode_layout = QVBoxLayout(mode_group)
        self._mode_btn_group = QButtonGroup(self)
        self._mode_btns: dict[EditMode, QRadioButton] = {}
        mode_defs = [
            (EditMode.BRUSH,           "ブラシ追加/削除 [B]"),
            (EditMode.RECT_ADD,        "矩形追加 [R]"),
            (EditMode.RECT_DEL,        "矩形削除 [Shift+R]"),
            (EditMode.POLY_ADD,        "ポリゴン追加 [P]"),
            (EditMode.POLY_DEL,        "ポリゴン削除 [Shift+P]"),
            (EditMode.GRABCUT_ADD,     "GrabCut有効化 [G]"),
            (EditMode.GRABCUT_DEL,     "GrabCut除外 [Shift+G]"),
            (EditMode.GRABCUT_REPLACE, "GrabCut置換 [Ctrl+G]"),
            (EditMode.PAN,             "パン操作"),
        ]
        for i, (mode, label) in enumerate(mode_defs):
            rb = QRadioButton(label)
            if mode == EditMode.BRUSH:
                rb.setChecked(True)
            self._mode_btn_group.addButton(rb, i)
            self._mode_btns[mode] = rb
            mode_layout.addWidget(rb)
        self._mode_btn_group.idClicked.connect(self._on_mode_btn_clicked)
        layout.addWidget(mode_group)

        # ----- GrabCut設定 -----
        self._grabcut_group = QGroupBox("GrabCut設定")
        grabcut_layout = QVBoxLayout(self._grabcut_group)

        grabcut_layout.addWidget(QLabel("反復回数 (1〜20):"))
        self._grabcut_iter_spin = QSpinBox()
        self._grabcut_iter_spin.setRange(1, 20)
        self._grabcut_iter_spin.setValue(5)
        self._grabcut_iter_spin.setToolTip("GrabCutの反復回数。大きいほど精度が上がるが遅くなる")
        self._grabcut_iter_spin.valueChanged.connect(self._canvas.set_grabcut_iter_count)
        grabcut_layout.addWidget(self._grabcut_iter_spin)

        self._grabcut_post_dilate_cb = QCheckBox("適用後に膨張")
        self._grabcut_post_dilate_cb.setChecked(False)
        self._grabcut_post_dilate_cb.toggled.connect(self._canvas.set_grabcut_post_dilate)
        grabcut_layout.addWidget(self._grabcut_post_dilate_cb)

        self._grabcut_post_erode_cb = QCheckBox("適用後に収縮")
        self._grabcut_post_erode_cb.setChecked(False)
        self._grabcut_post_erode_cb.toggled.connect(self._canvas.set_grabcut_post_erode)
        grabcut_layout.addWidget(self._grabcut_post_erode_cb)

        grabcut_layout.addWidget(QLabel("後処理カーネルサイズ:"))
        self._grabcut_post_kernel_spin = QSpinBox()
        self._grabcut_post_kernel_spin.setRange(1, 15)
        self._grabcut_post_kernel_spin.setValue(3)
        self._grabcut_post_kernel_spin.setSingleStep(2)
        self._grabcut_post_kernel_spin.valueChanged.connect(self._canvas.set_grabcut_post_kernel_size)
        grabcut_layout.addWidget(self._grabcut_post_kernel_spin)

        # 大画像設定
        grabcut_layout.addWidget(QLabel("─── 大画像最適化 ───"))
        self._grabcut_use_downscale_cb = QCheckBox("大画像を縮小して処理する")
        self._grabcut_use_downscale_cb.setChecked(True)
        self._grabcut_use_downscale_cb.setToolTip(
            "ONにすると大きな画像をROI切り出し+縮小してGrabCutを実行します。\n"
            "処理が速くなり、メモリ使用量も減ります。"
        )
        self._grabcut_use_downscale_cb.toggled.connect(self._canvas.set_grabcut_use_downscale)
        grabcut_layout.addWidget(self._grabcut_use_downscale_cb)

        grabcut_layout.addWidget(QLabel("GrabCut最大処理サイズ (px):"))
        self._grabcut_max_size_spin = QSpinBox()
        self._grabcut_max_size_spin.setRange(512, 4096)
        self._grabcut_max_size_spin.setValue(2048)
        self._grabcut_max_size_spin.setSingleStep(256)
        self._grabcut_max_size_spin.setToolTip("ROIの長辺がこのサイズを超えたら縮小して処理します")
        self._grabcut_max_size_spin.valueChanged.connect(self._canvas.set_grabcut_max_processing_size)
        grabcut_layout.addWidget(self._grabcut_max_size_spin)

        layout.addWidget(self._grabcut_group)

        # ----- 差分表示 -----
        diff_group = QGroupBox("差分表示")
        diff_layout = QVBoxLayout(diff_group)
        self._diff_cb = QCheckBox("差分表示 [F]")
        self._diff_cb.setChecked(False)
        self._diff_cb.toggled.connect(self._canvas.set_diff_mode)
        diff_layout.addWidget(QLabel("緑=追加 / 青=削除 / 赤=変化なし"))
        diff_layout.addWidget(self._diff_cb)
        layout.addWidget(diff_group)

        # ----- ブラシ設定 -----
        brush_group = QGroupBox("ブラシ設定")
        brush_layout = QVBoxLayout(brush_group)
        brush_layout.addWidget(QLabel("ブラシサイズ:"))
        self._brush_spin = QSpinBox()
        self._brush_spin.setRange(1, 300)
        self._brush_spin.setValue(20)
        self._brush_spin.valueChanged.connect(self._on_brush_size_changed)
        brush_layout.addWidget(self._brush_spin)
        self._brush_slider = QSlider(Qt.Orientation.Horizontal)
        self._brush_slider.setRange(1, 300)
        self._brush_slider.setValue(20)
        self._brush_slider.valueChanged.connect(self._on_brush_slider_changed)
        brush_layout.addWidget(self._brush_slider)
        layout.addWidget(brush_group)

        # ----- マスク表示 -----
        mask_group = QGroupBox("マスク表示")
        mask_layout = QVBoxLayout(mask_group)
        self._mask_visible_cb = QCheckBox("マスク表示 [M]")
        self._mask_visible_cb.setChecked(True)
        self._mask_visible_cb.toggled.connect(self._canvas.set_mask_visible)
        mask_layout.addWidget(self._mask_visible_cb)
        mask_layout.addWidget(QLabel("透明度:"))
        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(45)
        self._opacity_slider.valueChanged.connect(
            lambda v: self._canvas.set_mask_opacity(v / 100.0)
        )
        mask_layout.addWidget(self._opacity_slider)
        layout.addWidget(mask_group)

        # ----- モルフォロジー処理 -----
        morph_group = QGroupBox("モルフォロジー処理")
        morph_layout = QVBoxLayout(morph_group)

        dilate_row = QHBoxLayout()
        btn_d1 = QPushButton("膨張 +1")
        btn_d1.clicked.connect(lambda: self._apply_dilate(1))
        btn_d3 = QPushButton("膨張 +3")
        btn_d3.clicked.connect(lambda: self._apply_dilate(3))
        dilate_row.addWidget(btn_d1)
        dilate_row.addWidget(btn_d3)
        morph_layout.addLayout(dilate_row)

        erode_row = QHBoxLayout()
        btn_e1 = QPushButton("収縮 -1")
        btn_e1.clicked.connect(lambda: self._apply_erode(1))
        btn_e3 = QPushButton("収縮 -3")
        btn_e3.clicked.connect(lambda: self._apply_erode(3))
        erode_row.addWidget(btn_e1)
        erode_row.addWidget(btn_e3)
        morph_layout.addLayout(erode_row)

        morph_layout.addWidget(QLabel("穴埋めカーネルサイズ:"))
        close_row = QHBoxLayout()
        self._close_kernel_spin = QSpinBox()
        self._close_kernel_spin.setRange(1, 99)
        self._close_kernel_spin.setValue(5)
        self._close_kernel_spin.setSingleStep(2)
        close_row.addWidget(self._close_kernel_spin)
        btn_close = QPushButton("穴埋め")
        btn_close.clicked.connect(self._apply_close_holes)
        close_row.addWidget(btn_close)
        morph_layout.addLayout(close_row)

        layout.addWidget(morph_group)

        # ----- 小領域除去 -----
        comp_group = QGroupBox("小領域除去")
        comp_layout = QVBoxLayout(comp_group)
        comp_layout.addWidget(QLabel("面積閾値 (px):"))
        comp_row = QHBoxLayout()
        self._min_area_spin = QSpinBox()
        self._min_area_spin.setRange(1, 100000)
        self._min_area_spin.setValue(100)
        self._min_area_spin.setSingleStep(10)
        comp_row.addWidget(self._min_area_spin)
        btn_remove = QPushButton("小領域除去")
        btn_remove.clicked.connect(self._apply_remove_small)
        comp_row.addWidget(btn_remove)
        comp_layout.addLayout(comp_row)
        layout.addWidget(comp_group)

        # ----- 保存設定 -----
        save_group = QGroupBox("保存設定")
        save_layout = QVBoxLayout(save_group)
        self._colmap_cb = QCheckBox("保存時にCOLMAP互換\nマスクも出力する")
        self._colmap_cb.setChecked(False)
        self._colmap_cb.toggled.connect(lambda v: setattr(self, "_save_colmap", v))
        save_layout.addWidget(self._colmap_cb)
        layout.addWidget(save_group)

        # ----- 操作ボタン -----
        nav_group = QGroupBox("操作")
        nav_layout = QVBoxLayout(nav_group)

        self._btn_prev = QPushButton("← 前の画像 [A]")
        self._btn_prev.clicked.connect(self._prev_image)
        nav_layout.addWidget(self._btn_prev)

        self._btn_next = QPushButton("次の画像 → [D]")
        self._btn_next.clicked.connect(self._next_image)
        nav_layout.addWidget(self._btn_next)

        self._btn_save = QPushButton("保存 [S / Ctrl+S]")
        self._btn_save.setStyleSheet("QPushButton { background: #2a6; color: white; font-weight: bold; }")
        self._btn_save.clicked.connect(self._save_current)
        nav_layout.addWidget(self._btn_save)

        self._btn_undo = QPushButton("元に戻す [Z / Ctrl+Z]")
        self._btn_undo.clicked.connect(self._undo)
        nav_layout.addWidget(self._btn_undo)

        self._btn_redo = QPushButton("やり直し [Ctrl+Y]")
        self._btn_redo.clicked.connect(self._redo)
        nav_layout.addWidget(self._btn_redo)

        btn_resize = QPushButton("画像サイズに合わせてリサイズ")
        btn_resize.clicked.connect(self._resize_mask_to_image)
        btn_resize.setToolTip("マスクのサイズが画像と異なる場合に使用")
        nav_layout.addWidget(btn_resize)

        layout.addWidget(nav_group)

        # ----- 品質チェック -----
        check_group = QGroupBox("品質チェック")
        check_layout = QVBoxLayout(check_group)

        btn_bulk_check = QPushButton("一括チェック")
        btn_bulk_check.setStyleSheet("QPushButton { background: #46a; color: white; font-weight: bold; }")
        btn_bulk_check.setToolTip("全画像のマスク品質をチェックして一覧を更新")
        btn_bulk_check.clicked.connect(self._run_bulk_check)
        check_layout.addWidget(btn_bulk_check)

        btn_colmap_export = QPushButton("COLMAP互換出力")
        btn_colmap_export.setToolTip("元マスクから masks_colmap/ に一括出力")
        btn_colmap_export.clicked.connect(self._export_colmap_all)
        check_layout.addWidget(btn_colmap_export)

        btn_csv = QPushButton("ログCSV出力")
        btn_csv.setToolTip("mask_check_log.csv を出力")
        btn_csv.clicked.connect(self._export_check_log)
        check_layout.addWidget(btn_csv)

        layout.addWidget(check_group)

        # ----- マスク統計 -----
        stats_group = QGroupBox("マスク統計")
        stats_layout = QVBoxLayout(stats_group)
        stats_layout.setSpacing(2)

        def _stat_label() -> QLabel:
            lbl = QLabel("—")
            lbl.setWordWrap(True)
            lbl.setStyleSheet("font-size: 11px;")
            return lbl

        self._stat_image_size  = _stat_label()
        self._stat_mask_size   = _stat_label()
        self._stat_ratio       = _stat_label()
        self._stat_status      = _stat_label()
        self._stat_input_mask  = _stat_label()
        self._stat_edited_mask = _stat_label()
        self._stat_colmap_mask = _stat_label()

        for _caption, _stat_lbl in [
            ("画像サイズ:",    self._stat_image_size),
            ("マスクサイズ:",  self._stat_mask_size),
            ("マスク率:",      self._stat_ratio),
            ("状態:",          self._stat_status),
            ("入力マスク:",    self._stat_input_mask),
            ("編集済み:",      self._stat_edited_mask),
            ("COLMAPマスク:", self._stat_colmap_mask),
        ]:
            row_w = QWidget()
            row_l = QVBoxLayout(row_w)
            row_l.setContentsMargins(0, 0, 0, 0)
            row_l.setSpacing(0)
            cap = QLabel(_caption)
            cap.setStyleSheet("font-size: 10px; color: #aaa;")
            row_l.addWidget(cap)
            row_l.addWidget(_stat_lbl)
            stats_layout.addWidget(row_w)

        layout.addWidget(stats_group)

        # ----- 操作説明 -----
        help_group = QGroupBox("操作説明")
        help_layout = QVBoxLayout(help_group)
        help_text = QLabel(
            "左クリック: マスク追加(ブラシ)\n"
            "右クリック: マスク削除(ブラシ)\n"
            "中ボタン: パン\n"
            "ホイール: ズーム\n"
            "+/-: ブラシサイズ\n"
            "B: ブラシ  R: 矩形追加\n"
            "Shift+R: 矩形削除\n"
            "P: ポリゴン追加\n"
            "Shift+P: ポリゴン削除\n"
            "G: GrabCut有効化\n"
            "Shift+G: GrabCut除外\n"
            "Ctrl+G: GrabCut置換\n"
            "Enter: ポリゴン確定 / GrabCut適用\n"
            "Esc: キャンセル\n"
            "Backspace: 最後の頂点を削除\n"
            "F: 差分表示ON/OFF\n"
            "M: マスク表示ON/OFF\n"
            "S / Ctrl+S: 保存\n"
            "A / D: 前後の画像\n"
            "Z / Ctrl+Z: Undo\n"
            "Ctrl+Y: Redo"
        )
        help_text.setWordWrap(True)
        help_text.setStyleSheet("font-size: 11px; color: #aaa;")
        help_layout.addWidget(help_text)
        layout.addWidget(help_group)

        layout.addStretch()
        return widget

    def _setup_shortcuts(self) -> None:
        shortcuts = [
            ("S",         self._save_current),
            ("Ctrl+S",    self._save_current),
            ("A",         self._prev_image),
            ("D",         self._next_image),
            ("Z",         self._undo),
            ("Ctrl+Z",    self._undo),
            ("Ctrl+Y",    self._redo),
            ("M",         self._toggle_mask_visible),
            ("+",         self._brush_increase),
            ("=",         self._brush_increase),
            ("-",         self._brush_decrease),
            ("B",         lambda: self._set_mode(EditMode.BRUSH)),
            ("R",         lambda: self._set_mode(EditMode.RECT_ADD)),
            ("Shift+R",   lambda: self._set_mode(EditMode.RECT_DEL)),
            ("P",         lambda: self._set_mode(EditMode.POLY_ADD)),
            ("Shift+P",   lambda: self._set_mode(EditMode.POLY_DEL)),
            ("F",         self._toggle_diff),
            ("G",         lambda: self._set_mode(EditMode.GRABCUT_ADD)),
            ("Shift+G",   lambda: self._set_mode(EditMode.GRABCUT_DEL)),
            ("Ctrl+G",    lambda: self._set_mode(EditMode.GRABCUT_REPLACE)),
        ]
        for key_str, slot in shortcuts:
            action = QAction(self)
            action.setShortcut(QKeySequence(key_str))
            action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
            action.triggered.connect(slot)
            self.addAction(action)

    # ------------------------------------------------------------------ #
    # プロジェクト操作
    # ------------------------------------------------------------------ #

    def _open_project(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "プロジェクトフォルダを選択")
        if not folder:
            return
        self._load_project(Path(folder))

    def _load_project(self, root: Path) -> None:
        self._project = load_project(root)
        self._list_panel.set_entries(self._project.entries)
        self._current_index = -1
        self._editor = None
        self._canvas.clear()
        self._clear_stats_panel()

        n = len(self._project.entries)
        self.statusBar().showMessage(f"プロジェクト: {root}  |  画像: {n}枚")
        if n > 0:
            self._select_image(0)

    # ------------------------------------------------------------------ #
    # 画像選択・表示
    # ------------------------------------------------------------------ #

    def _on_image_selected(self, index: int) -> None:
        if index == self._current_index:
            return
        if self._has_unsaved():
            reply = QMessageBox.question(
                self, "未保存の変更",
                "未保存の変更があります。このまま移動しますか?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                self._list_panel.select_row(self._current_index)
                return
        self._select_image(index)

    def _select_image(self, index: int) -> None:
        if self._project is None or not (0 <= index < len(self._project.entries)):
            return

        entry = self._project.entries[index]
        self._current_index = index

        import cv2
        img = imread_jp(entry.image_path)
        if img is None:
            QMessageBox.warning(self, "エラー", f"画像を読み込めませんでした:\n{entry.image_path}")
            return

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        h, w = img.shape[:2]
        image_size = (w, h)

        mask, mismatch = load_mask_or_empty(entry.mask_path, image_size)
        entry.mask_size_mismatch = mismatch

        if mismatch:
            mh, mw = mask.shape[:2]
            QMessageBox.warning(
                self, "サイズ不一致",
                f"マスクサイズが画像と異なります。\n"
                f"画像: {w}x{h}\nマスク: {mw}x{mh}\n\n"
                "右パネルの「画像サイズに合わせてリサイズ」で修正できます。"
            )

        self._editor = MaskEditor(mask)
        self._canvas.set_image(img)
        self._canvas.set_editor(self._editor)

        self._list_panel.update_entry(index)
        self._update_title(entry)
        self._update_stats_panel(entry)

    def _has_unsaved(self) -> bool:
        if self._project is None or self._current_index < 0:
            return False
        return self._project.entries[self._current_index].is_modified

    def _update_title(self, entry: ImageEntry) -> None:
        modified = " *" if entry.is_modified else ""
        self.setWindowTitle(f"COLMAP Mask Editor v0.4A.1 - {entry.rel_path}{modified}")

    # ------------------------------------------------------------------ #
    # GrabCut Worker管理
    # ------------------------------------------------------------------ #

    def _on_grabcut_requested(self, info: dict) -> None:
        """キャンバスからGrabCutリクエストを受け取り、Workerスレッドを起動する。"""
        from core.grabcut_tool import GrabCutOptions
        from core.grabcut_worker import GrabCutWorker

        # 既存ワーカーが残っていればキャンセル
        if self._grabcut_worker is not None:
            self._grabcut_worker.request_cancel()
            self._cleanup_grabcut_worker()

        self._grabcut_request_id += 1
        request_id = self._grabcut_request_id
        self._grabcut_pending_mode = info["mode"]

        image: np.ndarray = info["image"]
        rect: tuple = info["rect"]
        options: GrabCutOptions = info["options"]

        _log.info(
            "GrabCutリクエスト受信: request_id=%d, rect=%s, mode=%s",
            request_id, rect, info["mode"],
        )

        worker = GrabCutWorker(image, rect, options, request_id)
        thread = QThread(self)

        req_id = request_id

        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        # QObject の bound method として接続することで、self (MainWindow) の
        # スレッド所属がメインスレッドと判定され、自動的に Queued 接続になる。
        # lambda/partial は thread affinity が不明で Direct 接続になる場合があるため使わない。
        worker.finished.connect(self._on_worker_finished)
        worker.failed.connect(self._on_worker_failed)
        worker.cancelled.connect(self._on_worker_cancelled)
        worker.progress.connect(self._on_grabcut_progress)
        # シグナル完了→スレッド終了→自動クリーンアップ (thread.wait() でブロックしない)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.cancelled.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        worker.cancelled.connect(worker.deleteLater)

        self._grabcut_worker = worker
        self._grabcut_thread = thread

        self._set_grabcut_ui_locked(True)
        self._show_grabcut_progress()

        thread.start()

    # ------------------------------------------------------------------
    # Worker シグナルの受信スロット (lambda を避け bound method で定義)
    # self が MainWindow (メインスレッド) のため Auto 接続が Queued になる
    # ------------------------------------------------------------------

    def _on_worker_finished(self) -> None:
        """worker.finished を受信。sender() でワーカーを特定して結果を取得。"""
        worker = self.sender()
        try:
            result = worker.result if worker is not None else None
            request_id = worker.request_id if worker is not None else -1
        except RuntimeError:
            result = None
            request_id = -1
        self._on_grabcut_finished(result, request_id)

    def _on_worker_failed(self, message: str) -> None:
        """worker.failed を受信。"""
        worker = self.sender()
        try:
            request_id = worker.request_id if worker is not None else self._grabcut_request_id
        except RuntimeError:
            request_id = self._grabcut_request_id
        self._on_grabcut_failed(message, request_id)

    def _on_worker_cancelled(self) -> None:
        """worker.cancelled を受信。"""
        worker = self.sender()
        try:
            request_id = worker.request_id if worker is not None else self._grabcut_request_id
        except RuntimeError:
            request_id = self._grabcut_request_id
        self._on_grabcut_cancelled(request_id)

    def _on_grabcut_finished(self, result: object, request_id: int) -> None:
        """Worker正常完了。リクエストIDが一致する場合のみプレビューに反映する。"""
        from core.grabcut_tool import GrabCutResult

        self._hide_grabcut_progress()
        self._set_grabcut_ui_locked(False)
        self._cleanup_grabcut_worker()

        if request_id != self._grabcut_request_id:
            _log.info("古いGrabCut結果を破棄 (request_id=%d, current=%d)",
                      request_id, self._grabcut_request_id)
            self._canvas.clear_grabcut_state()
            return

        gc_result: GrabCutResult = result
        self._canvas.set_grabcut_preview(gc_result.mask, self._grabcut_pending_mode)

        iw, ih = gc_result.original_size
        roi_x, roi_y, roi_w, roi_h = gc_result.roi
        pw, ph = gc_result.processing_size
        elapsed = gc_result.processing_time_sec
        scale = gc_result.scale

        status = (
            f"GrabCut完了: 元画像 {iw}x{ih} | "
            f"ROI {roi_w}x{roi_h} | "
            f"処理 {pw}x{ph} | "
            f"縮小率 {scale:.3f} | "
            f"処理時間 {elapsed:.2f}秒"
        )
        self.statusBar().showMessage(status, 8000)
        _log.info(status)

    def _on_grabcut_failed(self, message: str, request_id: int) -> None:
        """Workerエラー。UIを復元してエラーを表示する。"""
        self._hide_grabcut_progress()
        self._set_grabcut_ui_locked(False)
        self._canvas.clear_grabcut_state()
        self._cleanup_grabcut_worker()

        if request_id != self._grabcut_request_id:
            return  # 古いリクエストのエラーは無視

        _log.warning("GrabCutエラー: %s", message)
        QMessageBox.warning(self, "GrabCutエラー", message)

    def _on_grabcut_cancelled(self, request_id: int) -> None:
        """Workerキャンセル完了。UIを復元する。"""
        self._hide_grabcut_progress()
        self._set_grabcut_ui_locked(False)
        self._canvas.clear_grabcut_state()
        self._cleanup_grabcut_worker()

        if request_id == self._grabcut_request_id:
            self.statusBar().showMessage("GrabCutをキャンセルしました", 3000)

    def _on_grabcut_progress(self, message: str) -> None:
        """Worker進捗メッセージ。プログレスダイアログのラベルを更新する。"""
        if self._grabcut_progress_dlg is not None:
            self._grabcut_progress_dlg.setLabelText(f"GrabCut処理中...\n{message}")
        self.statusBar().showMessage(message)

    def _cancel_grabcut(self) -> None:
        """GrabCut処理のキャンセルを要求する。"""
        if self._grabcut_worker is not None:
            _log.info("GrabCutキャンセル要求")
            self._grabcut_worker.request_cancel()

    def _cleanup_grabcut_worker(self) -> None:
        """ワーカーとスレッドの参照を解放する。
        実際のクリーンアップは thread.finished → deleteLater で行われる。
        thread.wait() はメインスレッドをブロックするため使用しない。
        """
        if self._grabcut_thread is not None:
            if self._grabcut_thread.isRunning():
                self._grabcut_thread.quit()
            self._grabcut_thread = None
        self._grabcut_worker = None

    def _set_grabcut_ui_locked(self, locked: bool) -> None:
        """GrabCut処理中に操作を制限/解除する。"""
        enabled = not locked
        for rb in self._mode_btns.values():
            rb.setEnabled(enabled)
        self._grabcut_group.setEnabled(enabled)
        self._btn_prev.setEnabled(enabled)
        self._btn_next.setEnabled(enabled)
        self._btn_save.setEnabled(enabled)
        self._btn_undo.setEnabled(enabled)
        self._btn_redo.setEnabled(enabled)
        self._act_open.setEnabled(enabled)
        self._act_save.setEnabled(enabled)
        self._act_save_all.setEnabled(enabled)

    def _show_grabcut_progress(self) -> None:
        dlg = QProgressDialog("GrabCut処理中...", "キャンセル", 0, 0, self)
        dlg.setWindowTitle("GrabCut")
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setRange(0, 0)
        dlg.canceled.connect(self._cancel_grabcut)
        dlg.show()
        self._grabcut_progress_dlg = dlg

    def _hide_grabcut_progress(self) -> None:
        if self._grabcut_progress_dlg is not None:
            try:
                self._grabcut_progress_dlg.canceled.disconnect()
            except RuntimeError:
                pass
            self._grabcut_progress_dlg.close()
            self._grabcut_progress_dlg = None

    # ------------------------------------------------------------------ #
    # ウィンドウクローズ
    # ------------------------------------------------------------------ #

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._grabcut_worker is not None:
            _log.info("ウィンドウクローズ: GrabCutワーカーを停止します")
            self._grabcut_worker.request_cancel()
        if self._grabcut_thread is not None and self._grabcut_thread.isRunning():
            self._grabcut_thread.quit()
            self._grabcut_thread.wait(2000)  # クローズ時のみ短時間待機
        event.accept()

    # ------------------------------------------------------------------ #
    # マスク統計パネル
    # ------------------------------------------------------------------ #

    def _update_stats_panel(self, entry: ImageEntry) -> None:
        assert self._project is not None
        root = self._project.root
        cr = entry.check_result

        if cr is not None:
            self._stat_image_size.setText(f"{cr.image_width} x {cr.image_height}")
            self._stat_mask_size.setText(f"{cr.mask_width} x {cr.mask_height}" if cr.mask_readable else "—")
            self._stat_ratio.setText(f"{cr.mask_ratio * 100:.2f} %" if cr.mask_readable else "—")
            self._stat_status.setText(cr.status)
        else:
            if self._editor is not None:
                mh, mw = self._editor.mask.shape[:2]
                img = imread_jp(entry.image_path)
                if img is not None:
                    ih, iw = img.shape[:2]
                    self._stat_image_size.setText(f"{iw} x {ih}")
                else:
                    self._stat_image_size.setText("—")
                self._stat_mask_size.setText(f"{mw} x {mh}")
                total = mw * mh
                black = int(np.sum(self._editor.mask == 0))
                ratio = black / total * 100 if total > 0 else 0.0
                self._stat_ratio.setText(f"{ratio:.2f} %")
            else:
                self._stat_image_size.setText("—")
                self._stat_mask_size.setText("—")
                self._stat_ratio.setText("—")
            from ui.image_list_panel import get_entry_status
            self._stat_status.setText(get_entry_status(entry))

        def rel_str(p: Optional[Path]) -> str:
            if p is None:
                return "なし"
            try:
                return str(p.relative_to(root))
            except ValueError:
                return str(p)

        source_mask_path = get_source_mask_save_path(self._project.root, entry)
        self._stat_input_mask.setText(rel_str(source_mask_path) if source_mask_path.exists() else rel_str(entry.mask_path))
        edited_path = get_edited_mask_path(self._project.root, entry.rel_path)
        self._stat_edited_mask.setText(rel_str(edited_path) if edited_path.exists() else "なし")
        colmap_path = get_colmap_mask_path(self._project.root, entry.rel_path)
        self._stat_colmap_mask.setText(rel_str(colmap_path) if colmap_path.exists() else "なし")

    def _clear_stats_panel(self) -> None:
        for lbl in (
            self._stat_image_size, self._stat_mask_size, self._stat_ratio,
            self._stat_status, self._stat_input_mask, self._stat_edited_mask,
            self._stat_colmap_mask,
        ):
            lbl.setText("—")

    # ------------------------------------------------------------------ #
    # 編集モード
    # ------------------------------------------------------------------ #

    _MODE_ORDER = [
        EditMode.BRUSH,
        EditMode.RECT_ADD,
        EditMode.RECT_DEL,
        EditMode.POLY_ADD,
        EditMode.POLY_DEL,
        EditMode.GRABCUT_ADD,
        EditMode.GRABCUT_DEL,
        EditMode.GRABCUT_REPLACE,
        EditMode.PAN,
    ]

    def _on_mode_btn_clicked(self, btn_id: int) -> None:
        mode = self._MODE_ORDER[btn_id]
        self._canvas.set_edit_mode(mode)

    def _set_mode(self, mode: EditMode) -> None:
        self._canvas.set_edit_mode(mode)
        rb = self._mode_btns.get(mode)
        if rb:
            rb.setChecked(True)

    def _on_mode_changed(self, label: str) -> None:
        self._update_status_bar_mode(label)

    def _update_status_bar_mode(self, mode_label: str) -> None:
        msg = self.statusBar().currentMessage()
        if "|モード:" in msg:
            msg = msg[: msg.index("|モード:")]
        self.statusBar().showMessage(f"{msg.strip()}  |モード: {mode_label}")

    def _on_canvas_status_message(self, message: str, timeout: int) -> None:
        self.statusBar().showMessage(message, timeout)

    # ------------------------------------------------------------------ #
    # 差分表示
    # ------------------------------------------------------------------ #

    def _toggle_diff(self) -> None:
        self._diff_cb.setChecked(not self._diff_cb.isChecked())

    # ------------------------------------------------------------------ #
    # ブラシ操作
    # ------------------------------------------------------------------ #

    def _on_mask_changed(self) -> None:
        if self._project and 0 <= self._current_index < len(self._project.entries):
            entry = self._project.entries[self._current_index]
            entry.is_modified = True
            self._list_panel.update_entry(self._current_index)
            self._update_title(entry)

    def _on_brush_size_changed(self, value: int) -> None:
        self._brush_slider.blockSignals(True)
        self._brush_slider.setValue(value)
        self._brush_slider.blockSignals(False)
        self._canvas.set_brush_radius(value)

    def _on_brush_slider_changed(self, value: int) -> None:
        self._brush_spin.blockSignals(True)
        self._brush_spin.setValue(value)
        self._brush_spin.blockSignals(False)
        self._canvas.set_brush_radius(value)

    # ------------------------------------------------------------------ #
    # モルフォロジー処理
    # ------------------------------------------------------------------ #

    def _apply_morphology(self, new_mask: np.ndarray) -> None:
        if self._editor is None:
            return
        self._editor.mask[:] = new_mask
        self._on_mask_changed()
        self._canvas.update()

    def _apply_dilate(self, kernel_size: int) -> None:
        if self._editor is None:
            return
        from core.mask_morphology import dilate_mask
        self._editor.begin_stroke()
        new_mask = dilate_mask(self._editor.mask, kernel_size)
        self._apply_morphology(new_mask)
        self.statusBar().showMessage(f"膨張 +{kernel_size} を適用しました", 2000)

    def _apply_erode(self, kernel_size: int) -> None:
        if self._editor is None:
            return
        from core.mask_morphology import erode_mask
        self._editor.begin_stroke()
        new_mask = erode_mask(self._editor.mask, kernel_size)
        self._apply_morphology(new_mask)
        self.statusBar().showMessage(f"収縮 -{kernel_size} を適用しました", 2000)

    def _apply_close_holes(self) -> None:
        if self._editor is None:
            return
        from core.mask_morphology import close_holes
        ks = self._close_kernel_spin.value()
        self._editor.begin_stroke()
        new_mask = close_holes(self._editor.mask, ks)
        self._apply_morphology(new_mask)
        self.statusBar().showMessage(f"穴埋め (kernel={ks}) を適用しました", 2000)

    def _apply_remove_small(self) -> None:
        if self._editor is None:
            return
        from core.mask_components import remove_small_components
        min_area = self._min_area_spin.value()
        self._editor.begin_stroke()
        new_mask = remove_small_components(self._editor.mask, min_area)
        self._apply_morphology(new_mask)
        self.statusBar().showMessage(f"小領域除去 (面積<{min_area}px) を適用しました", 2000)

    # ------------------------------------------------------------------ #
    # ナビゲーション
    # ------------------------------------------------------------------ #

    def _prev_image(self) -> None:
        if self._project and self._current_index > 0:
            self._on_image_selected(self._current_index - 1)
            self._list_panel.select_row(self._current_index)

    def _next_image(self) -> None:
        if self._project and self._current_index < len(self._project.entries) - 1:
            self._on_image_selected(self._current_index + 1)
            self._list_panel.select_row(self._current_index)

    # ------------------------------------------------------------------ #
    # Undo / Redo
    # ------------------------------------------------------------------ #

    def _undo(self) -> None:
        if self._editor and self._editor.undo():
            self._on_mask_changed()
            self._canvas.update()

    def _redo(self) -> None:
        if self._editor and self._editor.redo():
            self._on_mask_changed()
            self._canvas.update()

    # ------------------------------------------------------------------ #
    # マスク表示
    # ------------------------------------------------------------------ #

    def _toggle_mask_visible(self) -> None:
        self._mask_visible_cb.setChecked(not self._mask_visible_cb.isChecked())

    # ------------------------------------------------------------------ #
    # ブラシサイズ
    # ------------------------------------------------------------------ #

    def _brush_increase(self) -> None:
        self._brush_spin.setValue(self._brush_spin.value() + 5)

    def _brush_decrease(self) -> None:
        self._brush_spin.setValue(max(1, self._brush_spin.value() - 5))

    # ------------------------------------------------------------------ #
    # リサイズ
    # ------------------------------------------------------------------ #

    def _resize_mask_to_image(self) -> None:
        if self._editor is None or self._project is None or self._current_index < 0:
            return
        entry = self._project.entries[self._current_index]
        img = imread_jp(entry.image_path)
        if img is None:
            return
        h, w = img.shape[:2]
        self._editor.resize_to(w, h)
        entry.mask_size_mismatch = False
        entry.is_modified = True
        self._list_panel.update_entry(self._current_index)
        self._update_title(entry)
        self._canvas.update()
        self.statusBar().showMessage(f"マスクを {w}x{h} にリサイズしました", 3000)

    # ------------------------------------------------------------------ #
    # 保存
    # ------------------------------------------------------------------ #

    def _save_source_mask(self, entry: ImageEntry, mask: np.ndarray) -> bool:
        assert self._project is not None
        save_path = get_source_mask_save_path(self._project.root, entry)
        if save_mask(mask, save_path):
            entry.is_modified = False
            entry.has_mask = True
            entry.mask_path = save_path
            if self._project.masks_dir is None:
                self._project.masks_dir = self._project.root / "masks"
            return True
        return False

    def _save_colmap_mask(self, entry: ImageEntry, mask: np.ndarray) -> bool:
        assert self._project is not None
        colmap_path = get_colmap_mask_path(self._project.root, entry.rel_path)
        return save_mask(mask, colmap_path)

    def _save_both_masks(self, entry: ImageEntry, mask: np.ndarray) -> bool:
        r1 = self._save_source_mask(entry, mask)
        r2 = self._save_colmap_mask(entry, mask)
        return r1 and r2

    def _save_current(self) -> None:
        if self._project is None or self._current_index < 0 or self._editor is None:
            return
        entry = self._project.entries[self._current_index]
        self._save_entry(entry, self._editor.mask)
        self._canvas.update_baseline()
        self.statusBar().showMessage(f"保存しました: {entry.rel_path}", 3000)

    def _save_all(self) -> None:
        if self._project is None:
            return
        saved = 0
        for i, entry in enumerate(self._project.entries):
            if entry.is_modified and i == self._current_index and self._editor is not None:
                self._save_entry(entry, self._editor.mask)
                saved += 1
        self._canvas.update_baseline()
        self.statusBar().showMessage(f"{saved} 枚を保存しました", 3000)

    def _save_entry(self, entry: ImageEntry, mask: np.ndarray) -> None:
        assert self._project is not None

        if self._save_colmap:
            ok = self._save_both_masks(entry, mask)
        else:
            ok = self._save_source_mask(entry, mask)

        if ok:
            idx = self._project.entries.index(entry)
            self._list_panel.update_entry(idx)
            self._update_title(entry)
            self._update_stats_panel(entry)

        self._write_log(entry, mask)

    def _write_log(self, entry: ImageEntry, mask: np.ndarray) -> None:
        assert self._project is not None
        log_path = self._project.root / "mask_edit_log.csv"
        write_header = not log_path.exists()

        mh, mw = mask.shape[:2]
        img = imread_jp(entry.image_path)
        iw, ih = (img.shape[1], img.shape[0]) if img is not None else (0, 0)

        save_path = get_source_mask_save_path(self._project.root, entry)
        row = {
            "image_path":       str(entry.image_path),
            "input_mask_path":  str(entry.mask_path) if entry.mask_path else "",
            "edited_mask_path": "",
            "saved_mask_path":  str(save_path),
            "status":           "saved",
            "width":            iw,
            "height":           ih,
            "mask_width":       mw,
            "mask_height":      mh,
            "timestamp":        datetime.datetime.now().isoformat(timespec="seconds"),
        }
        try:
            with open(log_path, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as e:
            _log.warning("ログ書き込みエラー: %s", e)

    # ------------------------------------------------------------------ #
    # 一括チェック
    # ------------------------------------------------------------------ #

    def _run_bulk_check(self) -> None:
        if self._project is None:
            QMessageBox.information(self, "情報", "プロジェクトを開いてください。")
            return

        from core.mask_checker import check_image

        total = len(self._project.entries)
        for entry in self._project.entries:
            entry.check_result = check_image(entry, self._project.root)

        self._list_panel.refresh_all()

        if 0 <= self._current_index < total:
            self._update_stats_panel(self._project.entries[self._current_index])

        from collections import Counter
        counts: Counter = Counter(e.check_result.status for e in self._project.entries if e.check_result)
        summary_lines = [f"  {s}: {n}枚" for s, n in sorted(counts.items())]
        summary = "\n".join(summary_lines)
        QMessageBox.information(
            self, "一括チェック完了",
            f"全 {total} 枚のチェックが完了しました。\n\n{summary}"
        )
        self.statusBar().showMessage(f"一括チェック完了: {total} 枚", 5000)

    # ------------------------------------------------------------------ #
    # COLMAP互換一括出力
    # ------------------------------------------------------------------ #

    def _export_colmap_all(self) -> None:
        if self._project is None:
            QMessageBox.information(self, "情報", "プロジェクトを開いてください。")
            return

        from core.colmap_export import export_all_colmap_masks

        ok_count, ng_count = export_all_colmap_masks(self._project)
        QMessageBox.information(
            self, "COLMAP互換出力完了",
            f"masks_colmap/ への出力が完了しました。\n\n"
            f"成功: {ok_count} 枚\n失敗（マスクなし含む）: {ng_count} 枚"
        )
        self.statusBar().showMessage(f"COLMAP互換出力完了: {ok_count} 枚", 5000)

        if self._project and 0 <= self._current_index < len(self._project.entries):
            self._update_stats_panel(self._project.entries[self._current_index])

    # ------------------------------------------------------------------ #
    # チェックログCSV出力
    # ------------------------------------------------------------------ #

    def _export_check_log(self) -> None:
        if self._project is None:
            QMessageBox.information(self, "情報", "プロジェクトを開いてください。")
            return

        unchecked = sum(1 for e in self._project.entries if e.check_result is None)
        if unchecked > 0:
            reply = QMessageBox.question(
                self, "チェック未実行",
                f"{unchecked} 枚が未チェックです。先に一括チェックを実行しますか?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._run_bulk_check()
                return

        from core.check_log import export_check_log

        log_path = export_check_log(self._project)
        QMessageBox.information(
            self, "CSV出力完了",
            f"チェックログを出力しました:\n{log_path}"
        )
        self.statusBar().showMessage(f"CSV出力: {log_path.name}", 5000)
