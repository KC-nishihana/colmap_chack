"""
メインウィンドウ: 全パネルの配置・操作統括・ショートカット・保存・ログ
"""

import csv
import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
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
from ui.image_canvas import ImageCanvas
from ui.image_list_panel import ImageListPanel


class MainWindow(QMainWindow):
    """アプリケーションのメインウィンドウ"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("COLMAP Mask Editor v0.2")
        self.resize(1440, 900)

        self._project: Optional[ProjectInfo] = None
        self._current_index: int = -1
        self._editor: Optional[MaskEditor] = None
        self._save_colmap: bool = False

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

        open_act = QAction("プロジェクトを開く(&O)...", self)
        open_act.setShortcut(QKeySequence("Ctrl+O"))
        open_act.triggered.connect(self._open_project)
        file_menu.addAction(open_act)

        save_act = QAction("保存(&S)", self)
        save_act.setShortcut(QKeySequence("Ctrl+S"))
        save_act.triggered.connect(self._save_current)
        file_menu.addAction(save_act)

        save_all_act = QAction("すべて保存(&A)", self)
        save_all_act.triggered.connect(self._save_all)
        file_menu.addAction(save_all_act)

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
        splitter.addWidget(self._canvas)

        # 右: コントロールパネル (スクロール対応)
        right_panel = self._build_right_panel()
        scroll = QScrollArea()
        scroll.setWidget(right_panel)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(210)
        scroll.setMaximumWidth(300)
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

        btn_prev = QPushButton("← 前の画像 [A]")
        btn_prev.clicked.connect(self._prev_image)
        nav_layout.addWidget(btn_prev)

        btn_next = QPushButton("次の画像 → [D]")
        btn_next.clicked.connect(self._next_image)
        nav_layout.addWidget(btn_next)

        btn_save = QPushButton("保存 [S / Ctrl+S]")
        btn_save.setStyleSheet("QPushButton { background: #2a6; color: white; font-weight: bold; }")
        btn_save.clicked.connect(self._save_current)
        nav_layout.addWidget(btn_save)

        btn_undo = QPushButton("元に戻す [Z / Ctrl+Z]")
        btn_undo.clicked.connect(self._undo)
        nav_layout.addWidget(btn_undo)

        btn_redo = QPushButton("やり直し [Ctrl+Y]")
        btn_redo.clicked.connect(self._redo)
        nav_layout.addWidget(btn_redo)

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

        self._stat_image_size = _stat_label()
        self._stat_mask_size  = _stat_label()
        self._stat_ratio      = _stat_label()
        self._stat_status     = _stat_label()
        self._stat_input_mask = _stat_label()
        self._stat_edited_mask = _stat_label()
        self._stat_colmap_mask = _stat_label()

        for _caption, _stat_lbl in [
            ("画像サイズ:", self._stat_image_size),
            ("マスクサイズ:", self._stat_mask_size),
            ("マスク率:", self._stat_ratio),
            ("状態:", self._stat_status),
            ("入力マスク:", self._stat_input_mask),
            ("編集済み:", self._stat_edited_mask),
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
            "左クリック: マスク追加\n"
            "右クリック: マスク削除\n"
            "中ボタン: パン\n"
            "ホイール: ズーム\n"
            "+/-: ブラシサイズ\n"
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
            ("S",       self._save_current),
            ("A",       self._prev_image),
            ("D",       self._next_image),
            ("Z",       self._undo),
            ("Ctrl+Z",  self._undo),
            ("Ctrl+Y",  self._redo),
            ("M",       self._toggle_mask_visible),
            ("+",       self._brush_increase),
            ("=",       self._brush_increase),
            ("-",       self._brush_decrease),
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
        self.setWindowTitle(f"COLMAP Mask Editor v0.2 - {entry.rel_path}{modified}")

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

        # パス表示（相対パス）
        def rel_str(p: Optional[Path]) -> str:
            if p is None:
                return "なし"
            try:
                return str(p.relative_to(root))
            except ValueError:
                return str(p)

        source_mask_path = get_source_mask_save_path(root, entry)
        self._stat_input_mask.setText(rel_str(source_mask_path) if source_mask_path.exists() else rel_str(entry.mask_path))
        edited_path = get_edited_mask_path(root, entry.rel_path)
        self._stat_edited_mask.setText(rel_str(edited_path) if edited_path.exists() else "なし")
        colmap_path = get_colmap_mask_path(root, entry.rel_path)
        self._stat_colmap_mask.setText(rel_str(colmap_path) if colmap_path.exists() else "なし")

    def _clear_stats_panel(self) -> None:
        for lbl in (
            self._stat_image_size, self._stat_mask_size, self._stat_ratio,
            self._stat_status, self._stat_input_mask, self._stat_edited_mask,
            self._stat_colmap_mask,
        ):
            lbl.setText("—")

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
    # 保存（機能10: 1関数に集約）
    # ------------------------------------------------------------------ #

    def _save_source_mask(self, entry: ImageEntry, mask: np.ndarray) -> bool:
        """Overwrite the source mask. Create project/masks/*.png if missing."""
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
        """masks_colmap/ に保存"""
        assert self._project is not None
        colmap_path = get_colmap_mask_path(self._project.root, entry.rel_path)
        return save_mask(mask, colmap_path)

    def _save_both_masks(self, entry: ImageEntry, mask: np.ndarray) -> bool:
        """Save the source mask and the COLMAP-compatible mask."""
        r1 = self._save_source_mask(entry, mask)
        r2 = self._save_colmap_mask(entry, mask)
        return r1 and r2

    def _save_current(self) -> None:
        if self._project is None or self._current_index < 0 or self._editor is None:
            return
        entry = self._project.entries[self._current_index]
        self._save_entry(entry, self._editor.mask)
        self.statusBar().showMessage(f"保存しました: {entry.rel_path}", 3000)

    def _save_all(self) -> None:
        if self._project is None:
            return
        saved = 0
        for i, entry in enumerate(self._project.entries):
            if entry.is_modified and i == self._current_index and self._editor is not None:
                self._save_entry(entry, self._editor.mask)
                saved += 1
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
        """保存操作のログをCSVに追記"""
        assert self._project is not None
        log_path = self._project.root / "mask_edit_log.csv"
        write_header = not log_path.exists()

        mh, mw = mask.shape[:2]
        img = imread_jp(entry.image_path)
        iw, ih = (img.shape[1], img.shape[0]) if img is not None else (0, 0)

        save_path = get_source_mask_save_path(self._project.root, entry)
        row = {
            "image_path":      str(entry.image_path),
            "input_mask_path": str(entry.mask_path) if entry.mask_path else "",
            "edited_mask_path": "",
            "saved_mask_path":  str(save_path),
            "status":          "saved",
            "width":           iw,
            "height":          ih,
            "mask_width":      mw,
            "mask_height":     mh,
            "timestamp":       datetime.datetime.now().isoformat(timespec="seconds"),
        }
        try:
            with open(log_path, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as e:
            print(f"[WARN] ログ書き込みエラー: {e}")

    # ------------------------------------------------------------------ #
    # 一括チェック（機能3）
    # ------------------------------------------------------------------ #

    def _run_bulk_check(self) -> None:
        if self._project is None:
            QMessageBox.information(self, "情報", "プロジェクトを開いてください。")
            return

        from core.mask_checker import check_image

        total = len(self._project.entries)
        for entry in self._project.entries:
            entry.check_result = check_image(entry, self._project.root)

        # 一覧を再構築（フィルタ再適用）
        self._list_panel.refresh_all()

        # 現在画像の統計を更新
        if 0 <= self._current_index < total:
            self._update_stats_panel(self._project.entries[self._current_index])

        # 統計サマリ
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
    # COLMAP互換一括出力（機能4）
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

        # 統計パネル更新
        if self._project and 0 <= self._current_index < len(self._project.entries):
            self._update_stats_panel(self._project.entries[self._current_index])

    # ------------------------------------------------------------------ #
    # チェックログCSV出力（機能6）
    # ------------------------------------------------------------------ #

    def _export_check_log(self) -> None:
        if self._project is None:
            QMessageBox.information(self, "情報", "プロジェクトを開いてください。")
            return

        # チェック未実行なら先に実行するか確認
        unchecked = sum(1 for e in self._project.entries if e.check_result is None)
        if unchecked > 0:
            reply = QMessageBox.question(
                self, "チェック未実行",
                f"{unchecked} 枚が未チェックです。先に一括チェックを実行しますか?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._run_bulk_check()
                return  # _run_bulk_check からの再呼び出しを想定しないので手動で続行
            # No の場合はそのままCSV出力

        from core.check_log import export_check_log

        log_path = export_check_log(self._project)
        QMessageBox.information(
            self, "CSV出力完了",
            f"チェックログを出力しました:\n{log_path}"
        )
        self.statusBar().showMessage(f"CSV出力: {log_path.name}", 5000)
