"""
メインウィンドウ: 全パネルの配置・操作統括・ショートカット・保存・ログ
v0.5.1: タブ化右パネル・QSettings設定保存・未確定GrabCut保護・Worker終了強化
"""

import csv
import datetime
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import QThread, QTimer, Qt
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
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.app_settings import AppSettings
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
from core.version import APP_DISPLAY_NAME
from ui.image_canvas import EditMode, GrabCutUiState, ImageCanvas
from ui.image_list_panel import ImageListPanel

_log = logging.getLogger(__name__)

# GrabCut UI状態ごとのステータスバー表示テキストと色
_GC_STATE_TEXT: "dict[GrabCutUiState, tuple[str, str]]" = {
    GrabCutUiState.IDLE:            ("GrabCut: 待機中",      "#aaa"),
    GrabCutUiState.INITIAL_RUNNING: ("GrabCut: 処理中...",   "#ffd700"),
    GrabCutUiState.PREVIEW:         ("GrabCut: プレビュー",  "#4af"),
    GrabCutUiState.HINT_EDITING:    ("GrabCut: ヒント編集",  "#4f8"),
    GrabCutUiState.REFINE_RUNNING:  ("GrabCut: 再推定中...", "#ffd700"),
}

# タブインデックス定数
_TAB_EDIT = 0
_TAB_GRABCUT = 1
_TAB_SAVE = 2


class MainWindow(QMainWindow):
    """アプリケーションのメインウィンドウ"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_DISPLAY_NAME)
        self.resize(1440, 900)

        self._project: Optional[ProjectInfo] = None
        self._current_index: int = -1
        self._editor: Optional[MaskEditor] = None
        self._save_colmap: bool = False

        # GrabCut Workerスレッド管理
        self._grabcut_thread: Optional[QThread] = None
        self._grabcut_worker = None           # GrabCutWorker (型循環回避)
        self._grabcut_request_id: int = 0    # リクエストID (インクリメント)
        self._grabcut_task_is_refine: bool = False
        self._grabcut_pending_mode: str = "add"
        self._grabcut_progress_dlg: Optional[QProgressDialog] = None

        # GrabCutSession (再推定用)
        self._gc_session = None  # GrabCutSession | None

        # 遅延クローズフラグ (Worker実行中にウィンドウを閉じようとした場合に使用)
        self._close_pending: bool = False

        # 設定管理
        self._app_settings = AppSettings()

        self._setup_menu()
        self._setup_central()
        self._setup_shortcuts()

        # GrabCut状態をステータスバー右端に常時表示
        self._gc_state_label = QLabel("GrabCut: 待機中")
        self._gc_state_label.setStyleSheet("font-size: 11px; color: #aaa; padding: 0 6px;")
        self.statusBar().addPermanentWidget(self._gc_state_label)

        # マスク統計の遅延更新タイマー (ブラシ連続描画中のちらつき防止)
        self._stats_refresh_timer = QTimer(self)
        self._stats_refresh_timer.setSingleShot(True)
        self._stats_refresh_timer.timeout.connect(self._refresh_stats_throttled)

        self.statusBar().showMessage("プロジェクトフォルダを開いてください  [File > Open Project]")

        # 設定を復元する
        self._restore_settings()
        _log.info("アプリ起動完了: %s", APP_DISPLAY_NAME)

    # ------------------------------------------------------------------ #
    # UI構築
    # ------------------------------------------------------------------ #

    def _setup_menu(self) -> None:
        menubar = self.menuBar()

        # --- ファイルメニュー ---
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

        # --- 設定メニュー ---
        settings_menu = menubar.addMenu("設定(&T)")

        reset_act = QAction("設定を初期化(&R)...", self)
        reset_act.triggered.connect(self._reset_settings)
        settings_menu.addAction(reset_act)

        # --- ヘルプメニュー ---
        help_menu = menubar.addMenu("ヘルプ(&H)")

        about_act = QAction("このアプリについて(&A)...", self)
        about_act.triggered.connect(self._show_about_dialog)
        help_menu.addAction(about_act)

    def _setup_central(self) -> None:
        self._main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左: 画像一覧
        self._list_panel = ImageListPanel()
        self._list_panel.setMinimumWidth(200)
        self._list_panel.setMaximumWidth(340)
        self._list_panel.image_selected.connect(self._on_image_selected)
        self._main_splitter.addWidget(self._list_panel)

        # 中央: キャンバス
        self._canvas = ImageCanvas()
        self._canvas.mask_changed.connect(self._on_mask_changed)
        self._canvas.mode_changed.connect(self._on_mode_changed)
        self._canvas.status_message.connect(self._on_canvas_status_message)
        self._canvas.grabcut_requested.connect(self._on_grabcut_requested)
        self._canvas.grabcut_refine_requested.connect(self._on_grabcut_refine_requested)
        self._canvas.grabcut_cancel_requested.connect(self._cancel_grabcut)
        self._canvas.grabcut_session_cancelled.connect(self._on_grabcut_session_cancelled)
        self._canvas.grabcut_state_changed.connect(self._on_grabcut_state_changed)
        self._main_splitter.addWidget(self._canvas)

        # 右: コントロールパネル (タブ + 常時表示エリア)
        right_container = self._build_right_container()
        self._main_splitter.addWidget(right_container)

        self._main_splitter.setStretchFactor(0, 0)
        self._main_splitter.setStretchFactor(1, 1)
        self._main_splitter.setStretchFactor(2, 0)

        self.setCentralWidget(self._main_splitter)

    def _build_right_container(self) -> QWidget:
        """右パネル: タブウィジェット + 常時表示ナビエリア。"""
        container = QWidget()
        container.setMinimumWidth(220)
        container.setMaximumWidth(340)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # タブウィジェット
        self._right_tab_widget = QTabWidget()
        self._right_tab_widget.setDocumentMode(False)

        tab0_scroll = QScrollArea()
        tab0_scroll.setWidgetResizable(True)
        tab0_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        tab0_scroll.setWidget(self._build_edit_tab())
        self._right_tab_widget.addTab(tab0_scroll, "編集")

        tab1_scroll = QScrollArea()
        tab1_scroll.setWidgetResizable(True)
        tab1_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        tab1_scroll.setWidget(self._build_grabcut_tab())
        self._right_tab_widget.addTab(tab1_scroll, "GrabCut")

        tab2_scroll = QScrollArea()
        tab2_scroll.setWidgetResizable(True)
        tab2_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        tab2_scroll.setWidget(self._build_save_tab())
        self._right_tab_widget.addTab(tab2_scroll, "保存・確認")

        layout.addWidget(self._right_tab_widget, stretch=1)

        # 常時表示: ナビゲーションボタン
        nav_widget = self._build_nav_area()
        layout.addWidget(nav_widget, stretch=0)

        return container

    def _build_edit_tab(self) -> QWidget:
        """編集タブ: 編集モード・ブラシ・マスク表示・差分・モルフォロジー・小領域除去。"""
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

        # ----- 差分表示 -----
        diff_group = QGroupBox("差分表示")
        diff_layout = QVBoxLayout(diff_group)
        self._diff_cb = QCheckBox("差分表示 [F]")
        self._diff_cb.setChecked(False)
        self._diff_cb.toggled.connect(self._canvas.set_diff_mode)
        diff_layout.addWidget(QLabel("緑=追加 / 青=削除 / 赤=変化なし"))
        diff_layout.addWidget(self._diff_cb)
        layout.addWidget(diff_group)

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

        layout.addStretch()
        return widget

    def _build_grabcut_tab(self) -> QWidget:
        """GrabCutタブ: GrabCut設定・補正・ヒント・状態表示。"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

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
        self._grabcut_post_kernel_spin.valueChanged.connect(
            self._canvas.set_grabcut_post_kernel_size
        )
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
        self._grabcut_max_size_spin.valueChanged.connect(
            self._canvas.set_grabcut_max_processing_size
        )
        grabcut_layout.addWidget(self._grabcut_max_size_spin)

        # 既存マスクを背景制約として使用
        self._grabcut_use_existing_mask_cb = QCheckBox("既存の除外領域を背景制約として使用")
        self._grabcut_use_existing_mask_cb.setChecked(False)
        self._grabcut_use_existing_mask_cb.setToolTip(
            "ONにすると現在マスクが0 (除外) の領域をGrabCutの背景制約として使用します。\n"
            "ROI内のみに適用されます。"
        )
        self._grabcut_use_existing_mask_cb.toggled.connect(
            self._canvas.set_grabcut_use_existing_mask_as_bgd
        )
        grabcut_layout.addWidget(self._grabcut_use_existing_mask_cb)
        layout.addWidget(self._grabcut_group)

        # ----- GrabCut補正 -----
        self._gc_correction_group = QGroupBox("GrabCut補正")
        gc_corr_layout = QVBoxLayout(self._gc_correction_group)

        gc_corr_layout.addWidget(QLabel("ヒント種別:"))

        hint_btn_row = QHBoxLayout()
        self._btn_hint_fg = QPushButton("対象ヒント")
        self._btn_hint_fg.setToolTip("この領域を必ず抽出対象として指定 (緑で描画)")
        self._btn_hint_fg.setStyleSheet(
            "QPushButton { background: #1a6; color: white; }"
            "QPushButton:checked { background: #0d0; color: black; border: 2px solid #0f0; font-weight: bold; }"
        )
        self._btn_hint_fg.clicked.connect(self._on_hint_fg_clicked)
        hint_btn_row.addWidget(self._btn_hint_fg)

        self._btn_hint_bg = QPushButton("背景ヒント")
        self._btn_hint_bg.setToolTip("この領域を必ず背景として指定 (赤で描画)")
        self._btn_hint_bg.setStyleSheet(
            "QPushButton { background: #a22; color: white; }"
            "QPushButton:checked { background: #f33; color: white; border: 2px solid #f44; font-weight: bold; }"
        )
        self._btn_hint_bg.clicked.connect(self._on_hint_bg_clicked)
        hint_btn_row.addWidget(self._btn_hint_bg)

        self._btn_hint_erase = QPushButton("ヒント消去")
        self._btn_hint_erase.setToolTip("この領域のヒントを消去して初回GrabCut状態に戻す")
        self._btn_hint_erase.setStyleSheet(
            "QPushButton:checked { background: #666; color: white; border: 2px solid #aaa; font-weight: bold; }"
        )
        self._btn_hint_erase.clicked.connect(self._on_hint_erase_clicked)
        hint_btn_row.addWidget(self._btn_hint_erase)
        gc_corr_layout.addLayout(hint_btn_row)

        # ヒントツールを排他的ボタングループとして管理
        self._hint_tool_group = QButtonGroup(self)
        self._hint_tool_group.setExclusive(True)
        for _hbtn in (self._btn_hint_fg, self._btn_hint_bg, self._btn_hint_erase):
            _hbtn.setCheckable(True)
            self._hint_tool_group.addButton(_hbtn)

        gc_corr_layout.addWidget(QLabel("ヒントブラシサイズ (1〜300px):"))
        hint_brush_row = QHBoxLayout()
        self._hint_radius_spin = QSpinBox()
        self._hint_radius_spin.setRange(1, 300)
        self._hint_radius_spin.setValue(20)
        self._hint_radius_spin.valueChanged.connect(self._on_hint_radius_changed)
        hint_brush_row.addWidget(self._hint_radius_spin)
        self._hint_radius_slider = QSlider(Qt.Orientation.Horizontal)
        self._hint_radius_slider.setRange(1, 300)
        self._hint_radius_slider.setValue(20)
        self._hint_radius_slider.valueChanged.connect(self._on_hint_radius_slider_changed)
        hint_brush_row.addWidget(self._hint_radius_slider)
        gc_corr_layout.addLayout(hint_brush_row)

        hint_history_row = QHBoxLayout()
        self._btn_hint_undo = QPushButton("ヒントUndo")
        self._btn_hint_undo.setToolTip("最後のヒントストロークを取り消す")
        self._btn_hint_undo.clicked.connect(self._canvas.gc_undo_hint)
        hint_history_row.addWidget(self._btn_hint_undo)

        self._btn_hint_redo = QPushButton("ヒントRedo")
        self._btn_hint_redo.setToolTip("取り消したヒントストロークをやり直す")
        self._btn_hint_redo.clicked.connect(self._canvas.gc_redo_hint)
        hint_history_row.addWidget(self._btn_hint_redo)

        self._btn_hint_clear = QPushButton("ヒント全消去")
        self._btn_hint_clear.setToolTip("全ヒントストロークを消去する [Ctrl+Shift+Z]")
        self._btn_hint_clear.clicked.connect(self._canvas.gc_clear_hints)
        hint_history_row.addWidget(self._btn_hint_clear)
        gc_corr_layout.addLayout(hint_history_row)

        self._btn_refine = QPushButton("再推定 [Ctrl+Enter]")
        self._btn_refine.setStyleSheet("QPushButton { background: #46a; color: white; font-weight: bold; }")
        self._btn_refine.setToolTip("現在のヒントを使ってGrabCutを再実行する")
        self._btn_refine.clicked.connect(self._canvas.request_grabcut_refine)
        gc_corr_layout.addWidget(self._btn_refine)

        apply_cancel_row = QHBoxLayout()
        self._btn_gc_apply = QPushButton("適用 [Enter]")
        self._btn_gc_apply.setStyleSheet(
            "QPushButton { background: #2a6; color: white; font-weight: bold; }"
        )
        self._btn_gc_apply.setToolTip("現在のGrabCutプレビューをマスクに適用する")
        self._btn_gc_apply.clicked.connect(self._canvas.apply_grabcut_preview)
        apply_cancel_row.addWidget(self._btn_gc_apply)

        self._btn_gc_cancel = QPushButton("キャンセル [Esc]")
        self._btn_gc_cancel.setToolTip("GrabCutセッションを破棄する")
        self._btn_gc_cancel.clicked.connect(self._canvas.cancel_grabcut_preview)
        apply_cancel_row.addWidget(self._btn_gc_cancel)
        gc_corr_layout.addLayout(apply_cancel_row)

        self._gc_correction_group.setEnabled(False)
        layout.addWidget(self._gc_correction_group)

        layout.addStretch()
        return widget

    def _build_save_tab(self) -> QWidget:
        """保存・確認タブ: 保存設定・品質チェック・COLMAP出力・CSV・統計・説明。"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # ----- 保存設定 -----
        save_group = QGroupBox("保存設定")
        save_layout = QVBoxLayout(save_group)
        self._colmap_cb = QCheckBox("保存時にCOLMAP互換\nマスクも出力する")
        self._colmap_cb.setChecked(False)
        self._colmap_cb.toggled.connect(lambda v: setattr(self, "_save_colmap", v))
        save_layout.addWidget(self._colmap_cb)
        layout.addWidget(save_group)

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
            "Ctrl+Enter: GrabCut再推定\n"
            "Esc: キャンセル\n"
            "Ctrl+Z: Undo / ヒントUndo\n"
            "Ctrl+Y: Redo / ヒントRedo\n"
            "Ctrl+Shift+Z: ヒント全消去\n"
            "Backspace: 最後の頂点を削除\n"
            "F: 差分表示ON/OFF\n"
            "M: マスク表示ON/OFF\n"
            "S / Ctrl+S: 保存\n"
            "A / D: 前後の画像\n"
        )
        help_text.setWordWrap(True)
        help_text.setStyleSheet("font-size: 11px; color: #aaa;")
        help_layout.addWidget(help_text)
        layout.addWidget(help_group)

        layout.addStretch()
        return widget

    def _build_nav_area(self) -> QWidget:
        """常時表示ナビゲーションエリア: 前後移動・保存・Undo/Redo・リサイズ。"""
        nav_group = QGroupBox("操作")
        nav_layout = QVBoxLayout(nav_group)
        nav_layout.setSpacing(4)

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

        return nav_group

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
    # 設定の保存・復元
    # ------------------------------------------------------------------ #

    def _restore_settings(self) -> None:
        """起動時に設定を復元する。"""
        s = self._app_settings

        # ウィンドウジオメトリ
        geom = s.load_bytes("window/geometry")
        if geom:
            self.restoreGeometry(geom)
        splitter_state = s.load_bytes("window/splitter_state")
        if splitter_state:
            self._main_splitter.restoreState(splitter_state)

        # タブ番号
        tab_idx = s.get("window/right_tab_index", 0)
        self._right_tab_widget.setCurrentIndex(tab_idx)

        # 編集設定
        brush_size = s.get("edit/brush_size", 20)
        self._brush_spin.setValue(brush_size)
        self._brush_slider.setValue(brush_size)
        self._canvas.set_brush_radius(brush_size)

        opacity = s.get("edit/mask_opacity", 45)
        self._opacity_slider.setValue(opacity)
        self._canvas.set_mask_opacity(opacity / 100.0)

        self._mask_visible_cb.setChecked(s.get("edit/mask_visible", True))
        self._diff_cb.setChecked(s.get("edit/diff_visible", False))
        self._close_kernel_spin.setValue(s.get("edit/morph_kernel_size", 5))
        self._min_area_spin.setValue(s.get("edit/min_area", 100))

        # GrabCut設定
        iter_count = s.get("grabcut/iter_count", 5)
        self._grabcut_iter_spin.setValue(iter_count)
        self._canvas.set_grabcut_iter_count(iter_count)

        self._grabcut_post_dilate_cb.setChecked(s.get("grabcut/post_dilate", False))
        self._grabcut_post_erode_cb.setChecked(s.get("grabcut/post_erode", False))
        self._grabcut_post_kernel_spin.setValue(s.get("grabcut/post_kernel_size", 3))

        use_downscale = s.get("grabcut/use_downscale", True)
        self._grabcut_use_downscale_cb.setChecked(use_downscale)
        self._canvas.set_grabcut_use_downscale(use_downscale)

        max_size = s.get("grabcut/max_size", 2048)
        self._grabcut_max_size_spin.setValue(max_size)
        self._canvas.set_grabcut_max_processing_size(max_size)

        self._grabcut_use_existing_mask_cb.setChecked(s.get("grabcut/use_existing_mask", False))

        hint_radius = s.get("grabcut/hint_radius", 20)
        self._hint_radius_spin.setValue(hint_radius)
        self._hint_radius_slider.setValue(hint_radius)
        self._canvas.set_hint_radius(hint_radius)

        _log.debug("設定を復元しました")

    def _save_settings(self) -> None:
        """終了時に設定を保存する。"""
        s = self._app_settings

        # ウィンドウジオメトリ
        s.save_bytes("window/geometry", self.saveGeometry())
        s.save_bytes("window/splitter_state", self._main_splitter.saveState())

        values = {
            "window/right_tab_index": self._right_tab_widget.currentIndex(),
            "edit/brush_size":        self._brush_spin.value(),
            "edit/mask_opacity":      self._opacity_slider.value(),
            "edit/mask_visible":      self._mask_visible_cb.isChecked(),
            "edit/diff_visible":      self._diff_cb.isChecked(),
            "edit/morph_kernel_size": self._close_kernel_spin.value(),
            "edit/min_area":          self._min_area_spin.value(),
            "grabcut/iter_count":     self._grabcut_iter_spin.value(),
            "grabcut/post_dilate":    self._grabcut_post_dilate_cb.isChecked(),
            "grabcut/post_erode":     self._grabcut_post_erode_cb.isChecked(),
            "grabcut/post_kernel_size": self._grabcut_post_kernel_spin.value(),
            "grabcut/use_downscale":  self._grabcut_use_downscale_cb.isChecked(),
            "grabcut/max_size":       self._grabcut_max_size_spin.value(),
            "grabcut/use_existing_mask": self._grabcut_use_existing_mask_cb.isChecked(),
            "grabcut/hint_radius":    self._hint_radius_spin.value(),
        }
        s.save(values)
        _log.debug("設定を保存しました")

    # ------------------------------------------------------------------ #
    # メニューアクション: 設定 / ヘルプ
    # ------------------------------------------------------------------ #

    def _reset_settings(self) -> None:
        """設定を初期化する。"""
        reply = QMessageBox.question(
            self,
            "設定の初期化",
            "保存されているアプリ設定を初期化しますか？\n次回起動時にデフォルト値へ戻ります。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            # ボタンラベルが "はい" ではなく "初期化" になるよう日本語ボタンで表示することが
            # 望ましいが、QMessageBox.question の簡易版ではボタンカスタマイズが難しいため
            # 標準の Yes/No を使用し、確認文言で意図を伝える。
            self._app_settings.reset()
            self.statusBar().showMessage("設定を初期化しました。次回起動時にデフォルト値が使われます。", 4000)
            _log.info("設定を初期化しました")

    def _show_about_dialog(self) -> None:
        """Aboutダイアログを表示する。"""
        QMessageBox.about(
            self,
            "このアプリについて",
            f"<b>{APP_DISPLAY_NAME}</b><br><br>"
            "COLMAP画像用マスクの確認・修正ツール<br><br>"
            "Python / PySide6 / OpenCV",
        )

    # ------------------------------------------------------------------ #
    # プロジェクト操作
    # ------------------------------------------------------------------ #

    def _open_project(self) -> None:
        last_folder = self._app_settings.get("file/last_folder", "")
        folder = QFileDialog.getExistingDirectory(
            self, "プロジェクトフォルダを選択", last_folder or ""
        )
        if not folder:
            return

        # 実行中Workerがあれば確認
        if not self._resolve_running_worker("プロジェクトを開く"):
            return
        # 未確定GrabCutセッションがあれば確認
        if not self._resolve_pending_grabcut_session("プロジェクトを開く"):
            return
        # 未保存マスクがあれば確認
        if not self._resolve_unsaved_mask("プロジェクトを開く"):
            return

        self._app_settings.set("file/last_folder", folder)
        self._app_settings.sync()
        self._load_project(Path(folder))
        _log.info("プロジェクト読込: %s", folder)

    def _load_project(self, root: Path) -> None:
        self._project = load_project(root)
        self._list_panel.set_entries(self._project.entries)
        self._current_index = -1
        self._editor = None
        self._gc_session = None
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
        # 1. 実行中Workerの解決
        if not self._resolve_running_worker("画像切替"):
            self._list_panel.select_row(self._current_index)
            return
        # 2. 未確定GrabCutSessionの解決
        if not self._resolve_pending_grabcut_session("画像切替"):
            self._list_panel.select_row(self._current_index)
            return
        # 3. 未保存通常マスクの解決
        if not self._resolve_unsaved_mask("画像切替"):
            self._list_panel.select_row(self._current_index)
            return
        # 4. 画像切替
        self._gc_session = None
        self._select_image(index)

    def _select_image(self, index: int) -> None:
        if self._project is None or not (0 <= index < len(self._project.entries)):
            return

        entry = self._project.entries[index]
        self._current_index = index

        _log.info("画像切替: index=%d, %s", index, entry.rel_path)

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
        self._gc_session = None
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
        self.setWindowTitle(f"{APP_DISPLAY_NAME} - {entry.rel_path}{modified}")

    # ------------------------------------------------------------------ #
    # 共通確認処理ヘルパー
    # ------------------------------------------------------------------ #

    def _resolve_running_worker(self, reason: str) -> bool:
        """GrabCut処理中の場合、確認ダイアログを出す。続行可能ならTrue、中止ならFalse。"""
        if self._grabcut_worker is None:
            return True

        box = QMessageBox(self)
        box.setWindowTitle("GrabCut処理中")
        box.setText(f"GrabCut処理中です。\n処理をキャンセルして {reason} を続行しますか？")
        cancel_btn = box.addButton("処理をキャンセル", QMessageBox.ButtonRole.AcceptRole)
        back_btn = box.addButton("戻る", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(back_btn)
        box.exec()

        if box.clickedButton() is cancel_btn:
            _log.info("GrabCutキャンセル要求 (reason=%s)", reason)
            self._grabcut_request_id += 1  # 飛んでくる結果を無効化
            self._grabcut_worker.request_cancel()
            self._hide_grabcut_progress()
            self._set_grabcut_ui_locked(False)
            self._canvas.clear_grabcut_state()
            self._gc_session = None
            # ワーカーを棄てて参照を解放 (スレッドは自力でクリーンアップする)
            if self._grabcut_thread is not None:
                self._grabcut_thread.quit()
                self._grabcut_thread = None
            self._grabcut_worker = None
            return True
        return False

    def _resolve_pending_grabcut_session(self, reason: str) -> bool:
        """
        GrabCutプレビューまたはヒント編集が残っている場合に確認する。
        続行可能ならTrue、操作を中止する場合はFalse。
        """
        state = self._canvas.gc_ui_state
        if state not in (GrabCutUiState.PREVIEW, GrabCutUiState.HINT_EDITING):
            return True

        result = self._ask_pending_grabcut()
        _log.info("GrabCut未確定確認結果: %s (reason=%s)", result, reason)

        if result == "apply":
            self._canvas.apply_grabcut_preview()
            self._gc_session = None
            return True
        elif result == "discard":
            self._canvas.cancel_grabcut_preview()
            self._gc_session = None
            return True
        else:
            return False

    def _ask_pending_grabcut(self) -> str:
        """GrabCut未確定ダイアログを表示し 'apply' / 'discard' / 'cancel' を返す。"""
        box = QMessageBox(self)
        box.setWindowTitle("GrabCut未確定")
        box.setText("GrabCutの未確定結果があります。")
        apply_btn  = box.addButton("適用",   QMessageBox.ButtonRole.AcceptRole)
        discard_btn = box.addButton("破棄",  QMessageBox.ButtonRole.DestructiveRole)
        cancel_btn  = box.addButton("キャンセル", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(cancel_btn)
        box.exec()
        clicked = box.clickedButton()
        if clicked is apply_btn:
            return "apply"
        if clicked is discard_btn:
            return "discard"
        return "cancel"

    def _resolve_unsaved_mask(self, reason: str) -> bool:
        """
        未保存マスクがある場合に確認する。
        続行可能ならTrue、中止ならFalse。
        """
        if not self._has_unsaved():
            return True

        result = self._ask_unsaved_mask()
        _log.info("未保存確認結果: %s (reason=%s)", result, reason)

        if result == "save":
            if self._project is None or self._current_index < 0 or self._editor is None:
                return False
            entry = self._project.entries[self._current_index]
            ok = self._save_entry(entry, self._editor.mask)
            self._canvas.update_baseline()
            if not ok:
                QMessageBox.warning(
                    self, "保存失敗",
                    f"マスクの保存に失敗しました:\n{entry.rel_path}\n\n"
                    "ディスクの空き容量やアクセス権限を確認してください。"
                )
                return False
            return True
        elif result == "discard":
            return True
        else:
            return False

    def _ask_unsaved_mask(self) -> str:
        """未保存マスク確認ダイアログを表示し 'save' / 'discard' / 'cancel' を返す。"""
        box = QMessageBox(self)
        box.setWindowTitle("未保存の変更")
        box.setText("この画像には未保存の変更があります。")
        save_btn    = box.addButton("保存",      QMessageBox.ButtonRole.AcceptRole)
        discard_btn = box.addButton("破棄",      QMessageBox.ButtonRole.DestructiveRole)
        cancel_btn  = box.addButton("キャンセル", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(save_btn)
        box.exec()
        clicked = box.clickedButton()
        if clicked is save_btn:
            return "save"
        if clicked is discard_btn:
            return "discard"
        return "cancel"

    # ------------------------------------------------------------------ #
    # GrabCut Worker管理 (初回)
    # ------------------------------------------------------------------ #

    def _on_grabcut_requested(self, info: dict) -> None:
        """キャンバスからGrabCutリクエストを受け取り、Workerスレッドを起動する。"""
        from core.grabcut_tool import GrabCutOptions
        from core.grabcut_worker import GrabCutTaskType, GrabCutWorker

        if self._grabcut_worker is not None:
            self._grabcut_worker.request_cancel()
            self._hide_grabcut_progress()
            self._cleanup_grabcut_worker()

        self._grabcut_request_id += 1
        request_id = self._grabcut_request_id
        self._grabcut_pending_mode = info["mode"]
        self._gc_session = None

        image: np.ndarray = info["image"]
        rect: tuple = info["rect"]
        options: GrabCutOptions = info["options"]
        current_mask: Optional[np.ndarray] = info.get("current_mask")

        _log.info(
            "GrabCutリクエスト受信: request_id=%d, rect=%s, mode=%s",
            request_id, rect, info["mode"],
        )

        worker = GrabCutWorker(
            image_bgr=image,
            rect=rect,
            options=options,
            request_id=request_id,
            task_type=GrabCutTaskType.INITIAL,
            current_mask=current_mask,
        )
        self._start_worker(worker)

    def _on_grabcut_refine_requested(self, info: dict) -> None:
        """キャンバスから再推定リクエストを受け取り、REFINEタスクWorkerを起動する。"""
        from core.grabcut_tool import GrabCutOptions, GrabCutSession
        from core.grabcut_worker import GrabCutTaskType, GrabCutWorker

        if self._gc_session is None:
            _log.warning("再推定リクエストがあったがGrabCutSessionがありません")
            self._canvas.clear_grabcut_state()
            return

        if self._grabcut_worker is not None:
            self._grabcut_worker.request_cancel()
            self._hide_grabcut_progress()
            self._cleanup_grabcut_worker()

        self._grabcut_request_id += 1
        request_id = self._grabcut_request_id

        strokes = info.get("strokes", [])
        options: GrabCutOptions = info.get("options")
        if options is None:
            options = GrabCutOptions(iter_count=2)

        session_copy = _copy_grabcut_session(self._gc_session)
        strokes_copy = list(strokes)

        _log.info(
            "GrabCut再推定リクエスト: request_id=%d, ストローク数=%d, 再推定=%d回目",
            request_id, len(strokes_copy), session_copy.refine_count + 1,
        )

        worker = GrabCutWorker(
            request_id=request_id,
            task_type=GrabCutTaskType.REFINE,
            session=session_copy,
            hint_strokes=strokes_copy,
            options=options,
        )
        self._start_worker(worker)

    def _start_worker(self, worker) -> None:
        """Workerをスレッドで起動し、シグナルを接続する。"""
        from core.grabcut_worker import GrabCutTaskType
        thread = QThread(self)

        worker.moveToThread(thread)
        thread.started.connect(worker.run)

        self._grabcut_task_is_refine = (worker._task_type == GrabCutTaskType.REFINE)
        worker.finished.connect(self._on_worker_finished)
        worker.failed.connect(self._on_worker_failed)
        worker.cancelled.connect(self._on_worker_cancelled)
        worker.progress.connect(self._on_grabcut_progress)

        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.cancelled.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)

        self._grabcut_worker = worker
        self._grabcut_thread = thread

        self._set_grabcut_ui_locked(True)
        self._show_grabcut_progress()

        _log.info("Worker開始: request_id=%d, refine=%s", self._grabcut_request_id, self._grabcut_task_is_refine)
        thread.start()

    # ------------------------------------------------------------------
    # Worker シグナルの受信スロット
    # ------------------------------------------------------------------

    def _on_worker_finished(self, request_id: int) -> None:
        """worker.finished(request_id) を受信。"""
        if request_id != self._grabcut_request_id:
            _log.info("古いWorker結果を破棄 (request_id=%d, current=%d)",
                      request_id, self._grabcut_request_id)
            return

        worker = self._grabcut_worker
        result = None
        session = None
        if worker is not None:
            try:
                result = worker.result
                session = worker.session
            except RuntimeError:
                _log.warning("finished ハンドラでワーカー属性にアクセス失敗 (request_id=%d)", request_id)

        if self._grabcut_task_is_refine:
            self._on_grabcut_refine_finished(session, request_id)
        else:
            self._on_grabcut_finished(result, session, request_id)

        _log.info("Worker完了: request_id=%d", request_id)
        self._check_deferred_close()

    def _on_worker_failed(self, message: str, request_id: int) -> None:
        """worker.failed(message, request_id) を受信。"""
        self._on_grabcut_failed(message, request_id, is_refine=self._grabcut_task_is_refine)
        _log.warning("Worker失敗: request_id=%d, message=%s", request_id, message)
        self._check_deferred_close()

    def _on_worker_cancelled(self, request_id: int) -> None:
        """worker.cancelled(request_id) を受信。"""
        self._on_grabcut_cancelled(request_id)
        _log.info("Workerキャンセル: request_id=%d", request_id)
        self._check_deferred_close()

    def _check_deferred_close(self) -> None:
        """遅延クローズが要求されていた場合に再度 close() を呼ぶ。"""
        if self._close_pending:
            self._close_pending = False
            self.close()

    def _on_grabcut_finished(self, result: object, session: object, request_id: int) -> None:
        """初回GrabCut Worker正常完了。"""
        from core.grabcut_tool import GrabCutResult, GrabCutSession

        if request_id != self._grabcut_request_id:
            return

        self._hide_grabcut_progress()
        self._set_grabcut_ui_locked(False)
        self._cleanup_grabcut_worker()

        if isinstance(session, GrabCutSession):
            self._gc_session = session

        if not isinstance(result, GrabCutResult):
            _log.error("GrabCut結果がGrabCutResultではありません: %s", type(result))
            self._canvas.clear_grabcut_state()
            QMessageBox.warning(self, "内部エラー", "GrabCut処理の結果が不正です。再試行してください。")
            return

        gc_result: GrabCutResult = result
        self._canvas.set_grabcut_preview(gc_result.mask, self._grabcut_pending_mode)

        # GrabCutタブへ自動切替
        self._right_tab_widget.setCurrentIndex(_TAB_GRABCUT)

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

    def _on_grabcut_refine_finished(self, session: object, request_id: int) -> None:
        """再推定Worker正常完了。"""
        from core.grabcut_tool import GrabCutSession

        if request_id != self._grabcut_request_id:
            return

        self._hide_grabcut_progress()
        self._set_grabcut_ui_locked(False)
        self._cleanup_grabcut_worker()

        if not isinstance(session, GrabCutSession):
            _log.error("再推定結果がGrabCutSessionではありません: %s", type(session))
            self._canvas.clear_grabcut_state()
            return

        self._gc_session = session
        self._canvas.update_grabcut_preview(session.preview_mask)

        # GrabCutタブへ自動切替
        self._right_tab_widget.setCurrentIndex(_TAB_GRABCUT)

        elapsed = session.processing_time_sec
        refine_count = session.refine_count
        status = (
            f"再推定完了 (第{refine_count}回): 処理時間 {elapsed:.2f}秒 | "
            "ヒント追加 or Enter=適用 / Esc=キャンセル"
        )
        self.statusBar().showMessage(status, 8000)

    def _on_grabcut_failed(self, message: str, request_id: int, is_refine: bool = False) -> None:
        """Workerエラー。UIを復元してエラーを表示する。"""
        if request_id != self._grabcut_request_id:
            return

        self._hide_grabcut_progress()
        self._set_grabcut_ui_locked(False)
        self._cleanup_grabcut_worker()

        if is_refine:
            self._canvas._set_gc_ui_state(GrabCutUiState.HINT_EDITING)
            QMessageBox.warning(self, "再推定エラー",
                                f"再推定に失敗しました。\n{message}\n\n"
                                "初回GrabCut結果のまま続けられます。")
        else:
            self._gc_session = None
            self._canvas.clear_grabcut_state()
            QMessageBox.warning(self, "GrabCutエラー", message)

    def _on_grabcut_cancelled(self, request_id: int) -> None:
        """Workerキャンセル完了。UIを復元する。"""
        if request_id != self._grabcut_request_id:
            return

        self._hide_grabcut_progress()
        self._set_grabcut_ui_locked(False)
        self._canvas.clear_grabcut_state()
        self._cleanup_grabcut_worker()
        self._gc_session = None
        self.statusBar().showMessage("GrabCutをキャンセルしました", 3000)

    def _on_grabcut_session_cancelled(self) -> None:
        """プレビューキャンセル (Esc or キャンセルボタン)。"""
        self._gc_session = None

    def _on_grabcut_progress(self, message: str) -> None:
        """Worker進捗メッセージ。"""
        if self._grabcut_progress_dlg is not None:
            self._grabcut_progress_dlg.setLabelText(f"GrabCut処理中...\n{message}")
        self.statusBar().showMessage(message)

    def _cancel_grabcut(self) -> None:
        """GrabCut処理のキャンセルを要求する。"""
        if self._grabcut_worker is not None:
            _log.info("GrabCutキャンセル要求")
            self._grabcut_worker.request_cancel()

    def _cleanup_grabcut_worker(self) -> None:
        """ワーカーとスレッドの参照を解放する。スレッドは quit() 後に自力クリーンアップする。"""
        if self._grabcut_thread is not None:
            if self._grabcut_thread.isRunning():
                self._grabcut_thread.quit()
            self._grabcut_thread = None
        if self._grabcut_worker is not None:
            self._grabcut_worker.deleteLater()
            self._grabcut_worker = None

    def _set_grabcut_ui_locked(self, locked: bool) -> None:
        """GrabCut処理中に操作を制限/解除する。"""
        enabled = not locked
        for rb in self._mode_btns.values():
            rb.setEnabled(enabled)
        for w in (
            self._grabcut_iter_spin,
            self._grabcut_post_dilate_cb,
            self._grabcut_post_erode_cb,
            self._grabcut_post_kernel_spin,
            self._grabcut_use_downscale_cb,
            self._grabcut_max_size_spin,
            self._grabcut_use_existing_mask_cb,
        ):
            w.setEnabled(enabled)
        self._btn_prev.setEnabled(enabled)
        self._btn_next.setEnabled(enabled)
        self._btn_save.setEnabled(enabled)
        self._btn_undo.setEnabled(enabled)
        self._btn_redo.setEnabled(enabled)
        self._act_open.setEnabled(enabled)
        self._act_save.setEnabled(enabled)
        self._act_save_all.setEnabled(enabled)
        self._gc_correction_group.setEnabled(enabled and self._gc_session is not None)

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

    def _on_grabcut_state_changed(self, state: GrabCutUiState) -> None:
        """キャンバスのGrabCut UI状態変化を受けてパネルを更新する。"""
        text, color = _GC_STATE_TEXT.get(state, ("GrabCut: 不明", "#aaa"))
        self._gc_state_label.setText(text)
        self._gc_state_label.setStyleSheet(f"font-size: 11px; color: {color}; padding: 0 6px;")

        has_session = self._gc_session is not None
        can_edit_hints = state in (GrabCutUiState.PREVIEW, GrabCutUiState.HINT_EDITING)

        self._gc_correction_group.setEnabled(can_edit_hints and has_session)

        hint_btns_enabled = can_edit_hints and has_session
        self._btn_hint_fg.setEnabled(hint_btns_enabled)
        self._btn_hint_bg.setEnabled(hint_btns_enabled)
        self._btn_hint_erase.setEnabled(hint_btns_enabled)
        self._btn_hint_undo.setEnabled(hint_btns_enabled)
        self._btn_hint_redo.setEnabled(hint_btns_enabled)
        self._btn_hint_clear.setEnabled(hint_btns_enabled)
        self._btn_refine.setEnabled(hint_btns_enabled)
        self._btn_gc_apply.setEnabled(can_edit_hints and has_session)
        self._btn_gc_cancel.setEnabled(can_edit_hints and has_session)

        if state == GrabCutUiState.IDLE:
            self._gc_correction_group.setEnabled(False)
            self._clear_hint_tool_buttons()

        if state == GrabCutUiState.HINT_EDITING:
            self._btn_undo.setToolTip("ヒントストロークを元に戻す [Ctrl+Z]")
            self._btn_redo.setToolTip("ヒントストロークをやり直す [Ctrl+Y]")
        else:
            self._btn_undo.setToolTip("マスク操作を元に戻す [Ctrl+Z / Z]")
            self._btn_redo.setToolTip("マスク操作をやり直す [Ctrl+Y]")

    # ------------------------------------------------------------------ #
    # ヒントツール操作
    # ------------------------------------------------------------------ #

    def _on_hint_fg_clicked(self) -> None:
        from core.grabcut_tool import GrabCutHintLabel
        self._canvas.set_hint_label(GrabCutHintLabel.FOREGROUND)
        self._btn_hint_fg.setChecked(True)

    def _on_hint_bg_clicked(self) -> None:
        from core.grabcut_tool import GrabCutHintLabel
        self._canvas.set_hint_label(GrabCutHintLabel.BACKGROUND)
        self._btn_hint_bg.setChecked(True)

    def _on_hint_erase_clicked(self) -> None:
        self._canvas.set_hint_label(None)
        self._btn_hint_erase.setChecked(True)

    def _clear_hint_tool_buttons(self) -> None:
        """GrabCutセッション終了時にヒントツールボタンの選択状態をリセットする。"""
        self._hint_tool_group.setExclusive(False)
        for btn in (self._btn_hint_fg, self._btn_hint_bg, self._btn_hint_erase):
            btn.setChecked(False)
        self._hint_tool_group.setExclusive(True)

    def _on_hint_radius_changed(self, value: int) -> None:
        self._hint_radius_slider.blockSignals(True)
        self._hint_radius_slider.setValue(value)
        self._hint_radius_slider.blockSignals(False)
        self._canvas.set_hint_radius(value)

    def _on_hint_radius_slider_changed(self, value: int) -> None:
        self._hint_radius_spin.blockSignals(True)
        self._hint_radius_spin.setValue(value)
        self._hint_radius_spin.blockSignals(False)
        self._canvas.set_hint_radius(value)

    # ------------------------------------------------------------------ #
    # ウィンドウクローズ
    # ------------------------------------------------------------------ #

    def closeEvent(self, event: QCloseEvent) -> None:
        # 遅延クローズの2回目 (Worker終了後に再呼び出しされた場合)
        if self._close_pending:
            self._save_settings()
            event.accept()
            _log.info("アプリ終了 (遅延クローズ完了)")
            return

        # 1. Worker実行中の確認
        if self._grabcut_worker is not None:
            box = QMessageBox(self)
            box.setWindowTitle("GrabCut処理中")
            box.setText("GrabCut処理中です。\n処理をキャンセルして終了しますか？")
            cancel_btn = box.addButton("処理をキャンセル", QMessageBox.ButtonRole.AcceptRole)
            back_btn   = box.addButton("戻る", QMessageBox.ButtonRole.RejectRole)
            box.setDefaultButton(back_btn)
            box.exec()
            if box.clickedButton() is cancel_btn:
                # Workerにキャンセルを要求し、終了は遅延させる
                self._close_pending = True
                event.ignore()
                self._grabcut_request_id += 1  # 結果を無効化
                self._grabcut_worker.request_cancel()
                _log.info("アプリ終了: GrabCutキャンセルを待機中")
                return
            else:
                event.ignore()
                return

        # 2. 未確定GrabCutSessionの確認
        if not self._resolve_pending_grabcut_session("終了"):
            event.ignore()
            return

        # 3. 未保存マスクの確認
        if not self._resolve_unsaved_mask("終了"):
            event.ignore()
            return

        self._save_settings()
        event.accept()
        _log.info("アプリ終了")

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
        self._stat_input_mask.setText(
            rel_str(source_mask_path) if source_mask_path.exists() else rel_str(entry.mask_path)
        )
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

    def _refresh_stats_throttled(self) -> None:
        """デバウンスタイマー発火時に統計パネルを更新する。"""
        if self._project and 0 <= self._current_index < len(self._project.entries):
            self._update_stats_panel(self._project.entries[self._current_index])

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
        self._auto_switch_tab_for_mode(mode)

    def _set_mode(self, mode: EditMode) -> None:
        self._canvas.set_edit_mode(mode)
        rb = self._mode_btns.get(mode)
        if rb:
            rb.setChecked(True)
        self._auto_switch_tab_for_mode(mode)

    def _auto_switch_tab_for_mode(self, mode: EditMode) -> None:
        """編集モードに応じてタブを自動切替する。"""
        if mode in (EditMode.GRABCUT_ADD, EditMode.GRABCUT_DEL, EditMode.GRABCUT_REPLACE):
            self._right_tab_widget.setCurrentIndex(_TAB_GRABCUT)
        elif mode in (
            EditMode.BRUSH, EditMode.RECT_ADD, EditMode.RECT_DEL,
            EditMode.POLY_ADD, EditMode.POLY_DEL,
        ):
            self._right_tab_widget.setCurrentIndex(_TAB_EDIT)

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
            self._stats_refresh_timer.start(500)

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
        """Undo: HINT_EDITING状態ではヒントUndo, それ以外は通常Undo。"""
        if self._canvas.gc_ui_state == GrabCutUiState.HINT_EDITING:
            self._canvas.gc_undo_hint()
        elif self._editor and self._editor.undo():
            self._on_mask_changed()
            self._canvas.update()
            self.statusBar().showMessage("マスク操作を元に戻しました", 2000)

    def _redo(self) -> None:
        """Redo: HINT_EDITING状態ではヒントRedo, それ以外は通常Redo。"""
        if self._canvas.gc_ui_state == GrabCutUiState.HINT_EDITING:
            self._canvas.gc_redo_hint()
        elif self._editor and self._editor.redo():
            self._on_mask_changed()
            self._canvas.update()
            self.statusBar().showMessage("マスク操作をやり直しました", 2000)

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
        ok = self._save_entry(entry, self._editor.mask)
        self._canvas.update_baseline()
        if ok:
            self.statusBar().showMessage(f"保存しました: {entry.rel_path}", 3000)
        else:
            QMessageBox.warning(
                self, "保存失敗",
                f"マスクの保存に失敗しました:\n{entry.rel_path}\n\n"
                "ディスクの空き容量やアクセス権限を確認してください。"
            )

    def _save_all(self) -> None:
        if self._project is None:
            return
        saved = 0
        failed = 0
        for i, entry in enumerate(self._project.entries):
            if entry.is_modified and i == self._current_index and self._editor is not None:
                ok = self._save_entry(entry, self._editor.mask)
                if ok:
                    saved += 1
                else:
                    failed += 1
        self._canvas.update_baseline()
        if failed > 0:
            QMessageBox.warning(
                self, "保存失敗",
                f"{failed} 枚の保存に失敗しました。\n"
                "ディスクの空き容量やアクセス権限を確認してください。"
            )
        self.statusBar().showMessage(f"{saved} 枚を保存しました", 3000)

    def _save_entry(self, entry: ImageEntry, mask: np.ndarray) -> bool:
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
        else:
            _log.error("マスク保存失敗: %s", entry.rel_path)

        self._write_log(entry, mask)
        return ok

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
        progress = QProgressDialog("マスクをチェック中...", "キャンセル", 0, total, self)
        progress.setWindowTitle("一括チェック")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(200)
        for i, entry in enumerate(self._project.entries):
            if progress.wasCanceled():
                break
            progress.setValue(i)
            entry.check_result = check_image(entry, self._project.root)
        progress.setValue(total)

        self._list_panel.refresh_all()

        if 0 <= self._current_index < total:
            self._update_stats_panel(self._project.entries[self._current_index])

        # 保存・確認タブへ自動切替
        self._right_tab_widget.setCurrentIndex(_TAB_SAVE)

        from collections import Counter
        counts: Counter = Counter(
            e.check_result.status for e in self._project.entries if e.check_result
        )
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


# ------------------------------------------------------------------ #
# ヘルパー関数
# ------------------------------------------------------------------ #

def _copy_grabcut_session(session) -> object:
    """GrabCutSessionをスレッド安全にコピーする。"""
    from core.grabcut_tool import GrabCutSession
    return GrabCutSession(
        original_size=session.original_size,
        original_rect=session.original_rect,
        roi=session.roi,
        processing_size=session.processing_size,
        scale=session.scale,
        was_downscaled=session.was_downscaled,
        roi_image_bgr=session.roi_image_bgr.copy(),
        base_label_mask=session.base_label_mask.copy(),
        label_mask=session.label_mask.copy(),
        bgd_model=session.bgd_model.copy(),
        fgd_model=session.fgd_model.copy(),
        preview_mask=session.preview_mask.copy(),
        processing_time_sec=session.processing_time_sec,
        refine_count=session.refine_count,
    )
