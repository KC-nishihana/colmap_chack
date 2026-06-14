"""
中央キャンバス: 画像表示・マスク半透明重ね・ブラシ/矩形/ポリゴン/GrabCut編集・差分表示・ズーム・パン
v0.4B: GrabCutUiState追加・対話型ヒント描画・再推定対応
"""

from enum import Enum, auto
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QCursor,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import QWidget

from core.mask_ops import MaskEditor


class EditMode(Enum):
    BRUSH = auto()
    RECT_ADD = auto()
    RECT_DEL = auto()
    POLY_ADD = auto()
    POLY_DEL = auto()
    GRABCUT_ADD = auto()
    GRABCUT_DEL = auto()
    GRABCUT_REPLACE = auto()
    AI_PROMPT = auto()   # v0.6: SAM2 プロンプト (正/負クリック・矩形)
    PAN = auto()


class GrabCutUiState(Enum):
    """GrabCut処理のUI状態遷移。"""
    IDLE = auto()             # セッションなし
    INITIAL_RUNNING = auto()  # 初回GrabCut実行中
    PREVIEW = auto()          # 初回結果表示中 (Enter=適用/Esc=キャンセル/ヒントボタンで HINT_EDITING へ)
    HINT_EDITING = auto()     # ヒント描画中
    REFINE_RUNNING = auto()   # 再推定実行中


_MODE_LABEL = {
    EditMode.BRUSH:          "ブラシ",
    EditMode.RECT_ADD:       "矩形追加",
    EditMode.RECT_DEL:       "矩形削除",
    EditMode.POLY_ADD:       "ポリゴン追加",
    EditMode.POLY_DEL:       "ポリゴン削除",
    EditMode.GRABCUT_ADD:    "GrabCut有効化",
    EditMode.GRABCUT_DEL:    "GrabCut除外",
    EditMode.GRABCUT_REPLACE:"GrabCut置換",
    EditMode.AI_PROMPT:      "AIプロンプト",
    EditMode.PAN:            "パン操作",
}

_GRABCUT_MODES = (EditMode.GRABCUT_ADD, EditMode.GRABCUT_DEL, EditMode.GRABCUT_REPLACE)


class ImageCanvas(QWidget):
    """
    画像とマスクを重ね表示し、各種編集・ズーム・パンを提供するウィジェット。
    v0.4B: GrabCut再推定用ヒント描画機能を追加。
    """

    mask_changed = Signal()
    mode_changed = Signal(str)           # 編集モード名
    status_message = Signal(str, int)    # メッセージ, タイムアウトms (0=持続)

    # GrabCut処理をメインウィンドウ側へ委譲するシグナル
    # dict: {"image": ndarray, "rect": tuple, "mode": str, "options": GrabCutOptions,
    #        "current_mask": ndarray|None}
    grabcut_requested = Signal(object)

    # 再推定リクエスト (ヒントストロークとオプションをMainWindowへ送る)
    # dict: {"strokes": list[HintStroke], "options": GrabCutOptions}
    grabcut_refine_requested = Signal(object)

    # キャンセル要求 (処理中にEscを押した場合)
    grabcut_cancel_requested = Signal()

    # セッションキャンセル (プレビュー中にEscを押した場合)
    grabcut_session_cancelled = Signal()

    # UI状態変化通知 (MainWindowがパネルの有効/無効を更新するために使用)
    grabcut_state_changed = Signal(object)  # GrabCutUiState

    # v0.6 AIプロンプト: クリック/矩形を元画像座標でMainWindowへ送る
    ai_point_clicked = Signal(object)  # dict {"x":int,"y":int,"positive":bool}
    ai_box_drawn = Signal(object)      # dict {"x1","y1","x2","y2"}

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

        # 表示対象
        self._image_bgr: Optional[np.ndarray] = None
        self._editor: Optional[MaskEditor] = None

        # 表示設定
        self._mask_opacity: float = 0.45
        self._mask_visible: bool = True
        self._mask_color: tuple[int, int, int] = (255, 60, 60)  # RGB

        # ブラシ設定
        self._brush_radius: int = 20
        self._brush_add: bool = True
        self._painting: bool = False
        self._stroke_started: bool = False

        # ズーム・パン
        self._scale: float = 1.0
        self._offset: QPointF = QPointF(0, 0)
        self._pan_last: Optional[QPoint] = None
        self._pan_active: bool = False

        # カーソル位置（マスク座標）
        self._cursor_pos: Optional[tuple[int, int]] = None

        # 編集モード
        self._edit_mode: EditMode = EditMode.BRUSH

        # 矩形編集
        self._rect_start: Optional[tuple[int, int]] = None
        self._rect_end: Optional[tuple[int, int]] = None
        self._rect_dragging: bool = False

        # ポリゴン編集
        self._poly_points: list[tuple[int, int]] = []

        # 差分表示
        self._diff_mode: bool = False
        self._baseline_mask: Optional[np.ndarray] = None

        # GrabCut設定
        self._grabcut_iter_count: int = 5
        self._grabcut_post_dilate: bool = False
        self._grabcut_post_erode: bool = False
        self._grabcut_post_kernel_size: int = 3
        self._grabcut_use_downscale: bool = True
        self._grabcut_max_processing_size: int = 2048
        self._grabcut_use_existing_mask_as_bgd: bool = False

        # GrabCutプレビュー状態 (V0.4A互換)
        self._grabcut_preview_mask: Optional[np.ndarray] = None
        self._grabcut_preview_mode: Optional[str] = None
        self._grabcut_rect: Optional[tuple[int, int, int, int]] = None

        # GrabCut UI状態 (V0.4B)
        self._gc_ui_state: GrabCutUiState = GrabCutUiState.IDLE

        # ヒント描画状態 (V0.4B)
        # label: GrabCutHintLabel.FOREGROUND/BACKGROUND or None (消去)
        self._gc_hint_label = None   # Optional[GrabCutHintLabel] - None=消去
        self._gc_hint_is_active: bool = False   # ヒントツールが選択されているか
        self._gc_hint_radius: int = 20
        self._gc_hint_strokes: list = []        # list[HintStroke]
        self._gc_hint_redo_stack: list = []     # list[HintStroke]

        # 現在描画中のストローク
        self._gc_hint_drawing: bool = False
        self._gc_current_stroke_pts: list[tuple[int, int]] = []

        # ----- v0.6 AIプロンプト表示状態 -----
        self._ai_active: bool = False                 # AI_PROMPTモードか
        self._ai_points: list[tuple[int, int, int]] = []  # (x, y, label) 元画像座標
        self._ai_box: Optional[tuple[int, int, int, int]] = None
        self._ai_preview_mask: Optional[np.ndarray] = None  # 選択候補マスク (H,W)
        # AI矩形ドラッグ
        self._ai_box_start: Optional[tuple[int, int]] = None
        self._ai_box_end: Optional[tuple[int, int]] = None
        self._ai_box_dragging: bool = False
        self._ai_press_widget_pos: Optional[QPointF] = None

        self.setMinimumSize(400, 300)

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #

    def set_image(self, image_bgr: np.ndarray) -> None:
        self._image_bgr = image_bgr
        self._editor = None
        self._reset_edit_state()
        self._fit_to_view()
        self.update()

    def set_editor(self, editor: MaskEditor) -> None:
        self._editor = editor
        self._baseline_mask = editor.mask.copy()
        self._reset_edit_state()
        self.update()

    def set_mask_opacity(self, opacity: float) -> None:
        self._mask_opacity = max(0.0, min(1.0, opacity))
        self.update()

    def set_mask_visible(self, visible: bool) -> None:
        self._mask_visible = visible
        self.update()

    def set_brush_radius(self, radius: int) -> None:
        self._brush_radius = max(1, radius)
        self.update()

    def get_brush_radius(self) -> int:
        return self._brush_radius

    def set_edit_mode(self, mode: EditMode) -> None:
        self._edit_mode = mode
        self._reset_edit_state()
        self.mode_changed.emit(_MODE_LABEL[mode])
        self.update()

    def get_edit_mode(self) -> EditMode:
        return self._edit_mode

    def set_diff_mode(self, enabled: bool) -> None:
        self._diff_mode = enabled
        self.update()

    def update_baseline(self) -> None:
        """保存後など、現在マスクを差分ベースラインに更新する"""
        if self._editor is not None:
            self._baseline_mask = self._editor.mask.copy()
        self.update()

    def clear(self) -> None:
        self._image_bgr = None
        self._editor = None
        self._baseline_mask = None
        self._reset_edit_state()
        self.update()

    # ----- GrabCut設定setter -----

    def set_grabcut_iter_count(self, value: int) -> None:
        self._grabcut_iter_count = max(1, min(20, value))

    def set_grabcut_post_dilate(self, enabled: bool) -> None:
        self._grabcut_post_dilate = enabled

    def set_grabcut_post_erode(self, enabled: bool) -> None:
        self._grabcut_post_erode = enabled

    def set_grabcut_post_kernel_size(self, value: int) -> None:
        self._grabcut_post_kernel_size = max(1, min(15, value))

    def set_grabcut_use_downscale(self, enabled: bool) -> None:
        self._grabcut_use_downscale = enabled

    def set_grabcut_max_processing_size(self, value: int) -> None:
        self._grabcut_max_processing_size = max(512, min(4096, value))

    def set_grabcut_use_existing_mask_as_bgd(self, enabled: bool) -> None:
        self._grabcut_use_existing_mask_as_bgd = enabled

    # ----- GrabCut状態 (読み取り) -----

    @property
    def grabcut_processing(self) -> bool:
        return self._gc_ui_state in (
            GrabCutUiState.INITIAL_RUNNING, GrabCutUiState.REFINE_RUNNING
        )

    @property
    def gc_ui_state(self) -> GrabCutUiState:
        return self._gc_ui_state

    # ----- GrabCutプレビューのセット/クリア (Workerから呼ばれる) -----

    def set_grabcut_preview(self, mask: np.ndarray, mode_str: str) -> None:
        """Worker処理完了後にメインウィンドウから呼ばれてプレビューをセットする。"""
        self._grabcut_preview_mask = mask
        self._grabcut_preview_mode = mode_str
        self._set_gc_ui_state(GrabCutUiState.PREVIEW)
        self.status_message.emit(
            "GrabCutプレビュー中: Enter=適用 / Esc=キャンセル / ヒントボタンで補正", 0
        )
        self.update()

    def update_grabcut_preview(self, mask: np.ndarray) -> None:
        """再推定完了後にメインウィンドウから呼ばれてプレビューを更新する。"""
        self._grabcut_preview_mask = mask
        self._set_gc_ui_state(GrabCutUiState.HINT_EDITING)
        self.status_message.emit(
            "再推定完了: Enter=適用 / Esc=キャンセル / ヒント追加 / Ctrl+Enter=再推定", 0
        )
        self.update()

    def clear_grabcut_state(self) -> None:
        """エラー・キャンセル時にプレビューと処理フラグをクリアする。"""
        self._grabcut_preview_mask = None
        self._grabcut_preview_mode = None
        self._grabcut_rect = None
        # ヒント状態もクリア
        self._gc_hint_strokes.clear()
        self._gc_hint_redo_stack.clear()
        self._gc_hint_drawing = False
        self._gc_current_stroke_pts.clear()
        self._gc_hint_is_active = False
        self._set_gc_ui_state(GrabCutUiState.IDLE)
        self.update()

    # ----- ヒントツール API (V0.4B) -----

    def set_hint_label(self, label) -> None:
        """
        ヒントラベルを設定してヒント描画モードに入る。
        label: GrabCutHintLabel.FOREGROUND / GrabCutHintLabel.BACKGROUND / None(消去)
        """
        self._gc_hint_label = label
        self._gc_hint_is_active = True
        if self._gc_ui_state == GrabCutUiState.PREVIEW:
            self._set_gc_ui_state(GrabCutUiState.HINT_EDITING)
            self.status_message.emit(
                "ヒント描画中: 左ドラッグ=描画 / 右クリック=取消 / Ctrl+Enter=再推定 / Enter=適用", 0
            )
        self.update()

    def deactivate_hint_tool(self) -> None:
        """ヒント描画モードを無効化する (GrabCut状態は維持)。"""
        self._gc_hint_is_active = False
        self.update()

    def set_hint_radius(self, radius: int) -> None:
        self._gc_hint_radius = max(1, min(300, radius))

    def get_hint_radius(self) -> int:
        return self._gc_hint_radius

    def gc_undo_hint(self) -> None:
        """最後のヒントストロークを取り消す。"""
        if self._gc_hint_strokes:
            stroke = self._gc_hint_strokes.pop()
            self._gc_hint_redo_stack.append(stroke)
            self.status_message.emit("ヒントを取り消しました", 2000)
            self.update()

    def gc_redo_hint(self) -> None:
        """取り消したヒントストロークをやり直す。"""
        if self._gc_hint_redo_stack:
            stroke = self._gc_hint_redo_stack.pop()
            self._gc_hint_strokes.append(stroke)
            self.status_message.emit("ヒントをやり直しました", 2000)
            self.update()

    def gc_clear_hints(self) -> None:
        """全ヒントストロークを消去する。"""
        if self._gc_hint_strokes or self._gc_hint_redo_stack:
            self._gc_hint_strokes.clear()
            self._gc_hint_redo_stack.clear()
            self.status_message.emit("ヒントを全消去しました", 2000)
            self.update()

    def get_hint_strokes(self) -> list:
        """現在のヒントストローク一覧のコピーを返す。"""
        return list(self._gc_hint_strokes)

    def request_grabcut_refine(self) -> None:
        """
        現在のヒントストロークを使ってGrabCut再推定をリクエストする。
        """
        if self._gc_ui_state not in (GrabCutUiState.PREVIEW, GrabCutUiState.HINT_EDITING):
            return

        from core.grabcut_tool import GrabCutOptions
        options = GrabCutOptions(
            iter_count=self._grabcut_iter_count,
            use_downscale=self._grabcut_use_downscale,
            max_processing_size=self._grabcut_max_processing_size,
        )

        self._set_gc_ui_state(GrabCutUiState.REFINE_RUNNING)
        self.grabcut_refine_requested.emit({
            "strokes": list(self._gc_hint_strokes),
            "options": options,
        })
        self.status_message.emit("GrabCut再推定中...", 0)
        self.update()

    # ------------------------------------------------------------------ #
    # v0.6 AIプロンプト オーバーレイ API (MainWindow が状態を渡す)
    # ------------------------------------------------------------------ #

    def set_ai_overlay(
        self,
        points: list,
        box,
        preview_mask: Optional[np.ndarray],
    ) -> None:
        """
        AIプロンプトと候補プレビューを表示する。
        points: list[(x, y, label)] / box: (x1,y1,x2,y2) or None / preview_mask: (H,W) uint8 or None
        座標はすべて元画像座標。
        """
        self._ai_points = list(points)
        self._ai_box = tuple(box) if box is not None else None
        self._ai_preview_mask = preview_mask
        self.update()

    def clear_ai_overlay(self) -> None:
        self._ai_points = []
        self._ai_box = None
        self._ai_preview_mask = None
        self._ai_box_start = None
        self._ai_box_end = None
        self._ai_box_dragging = False
        self.update()

    def set_ai_active(self, active: bool) -> None:
        """AI_PROMPTモードのアクティブ状態 (カーソル表示用)。"""
        self._ai_active = active
        self.update()

    # ------------------------------------------------------------------ #
    # 描画
    # ------------------------------------------------------------------ #

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(40, 40, 40))

        if self._image_bgr is None:
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "プロジェクトフォルダを開いてください")
            return

        # 表示解像度でコンポジットを生成 (フル解像度での処理を回避)
        disp_w = max(1, int(self._image_bgr.shape[1] * self._scale))
        disp_h = max(1, int(self._image_bgr.shape[0] * self._scale))
        composite = self._build_composite(disp_w, disp_h)
        qimg = _bgr_to_qimage(composite)
        pixmap = QPixmap.fromImage(qimg)

        painter.save()
        painter.translate(self._offset)
        painter.drawPixmap(0, 0, pixmap)
        painter.restore()

        # ヒントストローク描画 (コンポジットの上にQPainterで描く)
        if self._gc_ui_state in (GrabCutUiState.HINT_EDITING, GrabCutUiState.PREVIEW):
            for stroke in self._gc_hint_strokes:
                self._paint_hint_stroke(painter, stroke.points, stroke.radius, stroke.label)

        # 現在描画中のストローク
        if self._gc_hint_drawing and self._gc_current_stroke_pts:
            self._paint_hint_stroke(
                painter, self._gc_current_stroke_pts,
                self._gc_hint_radius, self._gc_hint_label,
                is_current=True,
            )

        # AIプロンプト: 矩形ドラッグ中プレビュー
        if (self._edit_mode == EditMode.AI_PROMPT and self._ai_box_dragging
                and self._ai_box_start and self._ai_box_end):
            x0, y0 = self._ai_box_start
            x1, y1 = self._ai_box_end
            wx0, wy0 = self._mask_to_widget(min(x0, x1), min(y0, y1))
            wx1, wy1 = self._mask_to_widget(max(x0, x1), max(y0, y1))
            painter.save()
            painter.setPen(QPen(QColor(255, 220, 0, 220), 1.5, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(wx0, wy0, wx1 - wx0, wy1 - wy0)
            painter.restore()

        # AIプロンプト: 確定済み矩形
        if self._ai_box is not None:
            x1, y1, x2, y2 = self._ai_box
            wx0, wy0 = self._mask_to_widget(min(x1, x2), min(y1, y2))
            wx1, wy1 = self._mask_to_widget(max(x1, x2), max(y1, y2))
            painter.save()
            painter.setPen(QPen(QColor(60, 140, 255, 230), 2.0, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(wx0, wy0, wx1 - wx0, wy1 - wy0)
            painter.restore()

        # AIプロンプト: 点マーカー (正=緑+ / 負=赤-)
        if self._ai_points:
            painter.save()
            for (px, py, label) in self._ai_points:
                wx, wy = self._mask_to_widget(px, py)
                if label == 1:
                    col = QColor(0, 210, 0)
                    sign = "+"
                else:
                    col = QColor(225, 30, 30)
                    sign = "-"
                painter.setPen(QPen(col, 2.0))
                painter.setBrush(QColor(col.red(), col.green(), col.blue(), 90))
                painter.drawEllipse(QPoint(wx, wy), 7, 7)
                painter.setPen(QPen(QColor(255, 255, 255), 2.0))
                painter.drawText(QRect(wx - 7, wy - 8, 14, 16),
                                 Qt.AlignmentFlag.AlignCenter, sign)
            painter.restore()

        # ブラシカーソル
        if self._edit_mode == EditMode.BRUSH and self._cursor_pos:
            mx, my = self._cursor_pos
            cx = int(mx * self._scale + self._offset.x())
            cy = int(my * self._scale + self._offset.y())
            r = int(self._brush_radius * self._scale)
            painter.save()
            if self._brush_add:
                painter.setPen(QPen(QColor(255, 255, 255, 200), 1.5))
            else:
                painter.setPen(QPen(QColor(100, 200, 255, 200), 1.5))
            painter.drawEllipse(QPoint(cx, cy), r, r)
            painter.restore()

        # ヒントブラシカーソル (ヒントモードでドラッグしていない時)
        if (self._gc_hint_is_active and not self._gc_hint_drawing and
                self._cursor_pos is not None and
                self._gc_ui_state in (GrabCutUiState.HINT_EDITING, GrabCutUiState.PREVIEW)):
            mx, my = self._cursor_pos
            cx = int(mx * self._scale + self._offset.x())
            cy = int(my * self._scale + self._offset.y())
            r = max(1, int(self._gc_hint_radius * self._scale))
            painter.save()
            if self._gc_hint_label is None:
                # 消去カーソル
                painter.setPen(QPen(QColor(220, 220, 220, 200), 1.5, Qt.PenStyle.DashLine))
            else:
                from core.grabcut_tool import GrabCutHintLabel
                if self._gc_hint_label == GrabCutHintLabel.FOREGROUND:
                    painter.setPen(QPen(QColor(0, 220, 0, 220), 2.0))
                else:
                    painter.setPen(QPen(QColor(220, 0, 0, 220), 2.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPoint(cx, cy), r, r)
            painter.restore()

        # 矩形プレビュー (RECT_ADD/DEL + GrabCut系モード)
        if self._rect_dragging and self._rect_start and self._rect_end:
            x0, y0 = self._rect_start
            x1, y1 = self._rect_end
            wx0 = int(min(x0, x1) * self._scale + self._offset.x())
            wy0 = int(min(y0, y1) * self._scale + self._offset.y())
            wx1 = int(max(x0, x1) * self._scale + self._offset.x())
            wy1 = int(max(y0, y1) * self._scale + self._offset.y())
            painter.save()
            if self._edit_mode in (EditMode.RECT_ADD, EditMode.GRABCUT_ADD, EditMode.GRABCUT_REPLACE):
                fill = QColor(255, 255, 255, 60)
                border = QColor(255, 255, 255, 200)
            else:  # RECT_DEL or GRABCUT_DEL
                fill = QColor(0, 120, 255, 60)
                border = QColor(0, 120, 255, 200)
            painter.fillRect(wx0, wy0, wx1 - wx0, wy1 - wy0, fill)
            painter.setPen(QPen(border, 1.5, Qt.PenStyle.DashLine))
            painter.drawRect(wx0, wy0, wx1 - wx0, wy1 - wy0)
            painter.restore()

        # ポリゴンプレビュー
        if self._edit_mode in (EditMode.POLY_ADD, EditMode.POLY_DEL) and self._poly_points:
            painter.save()
            if self._edit_mode == EditMode.POLY_ADD:
                color = QColor(255, 255, 100, 220)
            else:
                color = QColor(100, 180, 255, 220)
            pen = QPen(color, 2.0)
            painter.setPen(pen)
            pts_w = [self._mask_to_widget(px, py) for px, py in self._poly_points]
            for i in range(len(pts_w) - 1):
                painter.drawLine(pts_w[i][0], pts_w[i][1], pts_w[i + 1][0], pts_w[i + 1][1])
            # カーソルへのライン
            if self._cursor_pos:
                cx2, cy2 = self._mask_to_widget(*self._cursor_pos)
                last = pts_w[-1]
                painter.setPen(QPen(color, 1.5, Qt.PenStyle.DashLine))
                painter.drawLine(last[0], last[1], cx2, cy2)
            # 頂点ドット
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            for px, py in pts_w:
                painter.drawEllipse(QPoint(px, py), 4, 4)
            painter.restore()

        # 処理中インジケーター
        if self._gc_ui_state in (GrabCutUiState.INITIAL_RUNNING, GrabCutUiState.REFINE_RUNNING):
            painter.save()
            painter.setPen(QColor(255, 200, 0))
            label = "GrabCut処理中..." if self._gc_ui_state == GrabCutUiState.INITIAL_RUNNING else "再推定中..."
            painter.drawText(10, 20, label)
            painter.restore()

    def _paint_hint_stroke(
        self, painter: QPainter,
        pts: list[tuple[int, int]],
        radius: int,
        label,
        is_current: bool = False,
    ) -> None:
        """ヒントストロークをQPainterで描画する。"""
        if not pts:
            return

        from core.grabcut_tool import GrabCutHintLabel

        if label == GrabCutHintLabel.FOREGROUND:
            stroke_color = QColor(0, 220, 0, 200 if not is_current else 255)
        elif label == GrabCutHintLabel.BACKGROUND:
            stroke_color = QColor(220, 0, 0, 200 if not is_current else 255)
        else:  # 消去
            stroke_color = QColor(220, 220, 220, 180 if not is_current else 220)

        r = max(1, int(radius * self._scale))
        pen_width = max(1, r * 2)

        painter.save()
        if label is None:  # 消去
            pen = QPen(stroke_color, pen_width, Qt.PenStyle.DashLine,
                       Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        else:
            pen = QPen(stroke_color, pen_width, Qt.PenStyle.SolidLine,
                       Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        pts_w = [self._mask_to_widget(x, y) for x, y in pts]

        for i in range(len(pts_w) - 1):
            x0, y0 = pts_w[i]
            x1, y1 = pts_w[i + 1]
            painter.drawLine(QPoint(x0, y0), QPoint(x1, y1))

        # 最初と最後に円を描いて端点をきれいに
        if len(pts_w) == 1:
            x0, y0 = pts_w[0]
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(stroke_color)
            painter.drawEllipse(QPoint(x0, y0), r, r)

        painter.restore()

    def _build_composite(self, disp_w: int = 0, disp_h: int = 0) -> np.ndarray:
        """コンポジット画像を生成する。
        disp_w/disp_h が指定された場合はその解像度で処理し、大画像の描画コストを削減する。
        """
        ih, iw = self._image_bgr.shape[:2]

        # 表示解像度へダウンスケール (拡大時はそのまま)
        if disp_w > 0 and disp_h > 0 and (disp_w != iw or disp_h != ih):
            interp = cv2.INTER_LINEAR if disp_w <= iw else cv2.INTER_CUBIC
            img = cv2.resize(self._image_bgr, (disp_w, disp_h), interpolation=interp)
        else:
            img = self._image_bgr.copy()

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        dh, dw = img.shape[:2]
        needs_scale = (dw != iw or dh != ih)

        result = img
        if self._mask_visible and self._editor is not None:
            mask = self._editor.mask
            if needs_scale:
                mask = cv2.resize(mask, (dw, dh), interpolation=cv2.INTER_NEAREST)

            if self._diff_mode and self._baseline_mask is not None and self._editor.mask.shape == self._baseline_mask.shape:
                base_mask = self._baseline_mask
                if needs_scale:
                    base_mask = cv2.resize(base_mask, (dw, dh), interpolation=cv2.INTER_NEAREST)
                result = self._build_diff_composite(img, mask, base_mask)
            elif mask.shape[:2] == img.shape[:2]:
                overlay = img.copy()
                r, g, b = self._mask_color
                overlay[mask == 255] = [b, g, r]  # BGR
                result = cv2.addWeighted(overlay, self._mask_opacity, img, 1.0 - self._mask_opacity, 0)

        # GrabCutプレビューをマスク表示ON/OFFに関わらず重ねる
        if self._grabcut_preview_mask is not None and self._grabcut_preview_mask.shape[:2] == (ih, iw):
            gc_mask = self._grabcut_preview_mask
            if needs_scale:
                gc_mask = cv2.resize(gc_mask, (dw, dh), interpolation=cv2.INTER_NEAREST)
            result = self._overlay_grabcut_preview(result, gc_mask)

        # AI候補プレビュー (マスク表示ON/OFFに関わらず重ねる)
        if self._ai_preview_mask is not None and self._ai_preview_mask.shape[:2] == (ih, iw):
            ai_mask = self._ai_preview_mask
            if needs_scale:
                ai_mask = cv2.resize(ai_mask, (dw, dh), interpolation=cv2.INTER_NEAREST)
            result = self._overlay_ai_preview(result, ai_mask)

        return result

    def _overlay_ai_preview(self, base: np.ndarray, ai_mask: np.ndarray) -> np.ndarray:
        """AI候補マスクをシアン系の半透明オーバーレイで表示。"""
        region = ai_mask >= 128
        if not np.any(region):
            return base
        color = np.array([230, 200, 0], dtype=np.float32)  # BGR: シアン寄り
        result = base.copy().astype(np.float32)
        alpha = 0.5
        result[region] = result[region] * (1.0 - alpha) + color * alpha
        return result.clip(0, 255).astype(np.uint8)

    def _build_diff_composite(self, img: np.ndarray, mask: np.ndarray,
                               base_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """差分表示: 追加=緑, 削除=青, 変化なし=赤半透明"""
        if base_mask is None:
            base_mask = self._baseline_mask
        result = img.copy()

        added   = (mask == 255) & (base_mask == 0)    # 追加領域: 緑
        removed = (mask == 0)   & (base_mask == 255)  # 削除領域: 青
        kept    = (mask == 255) & (base_mask == 255)  # 変化なし: 赤半透明

        overlay = result.copy()
        overlay[added]   = (0,   200, 0)    # BGR: 緑
        overlay[removed] = (200, 0,   0)    # BGR: 青
        overlay[kept]    = (0,   0,   180)  # BGR: 赤

        changed_mask = added | removed | kept
        alpha = self._mask_opacity
        result[changed_mask] = cv2.addWeighted(
            overlay, alpha, result, 1.0 - alpha, 0
        )[changed_mask]

        return result

    def _overlay_grabcut_preview(self, base: np.ndarray,
                                  gc_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """GrabCutプレビュー結果を半透明オーバーレイで表示"""
        if gc_mask is None:
            gc_mask = self._grabcut_preview_mask
        if gc_mask is None:
            return base

        region = gc_mask == 255
        if not np.any(region):
            return base

        # モードに応じた色 (BGR)
        mode = self._grabcut_preview_mode
        if mode == "add":
            color = np.array([0, 220, 220], dtype=np.float32)    # 黄色 (GrabCut有効化)
        elif mode == "remove":
            color = np.array([200, 80, 0], dtype=np.float32)     # 青系 (GrabCut除外)
        else:  # replace
            color = np.array([50, 200, 80], dtype=np.float32)    # 緑系 (GrabCut置換)

        result = base.copy().astype(np.float32)
        alpha = 0.55
        result[region] = result[region] * (1.0 - alpha) + color * alpha
        return result.clip(0, 255).astype(np.uint8)

    # ------------------------------------------------------------------ #
    # マウス操作
    # ------------------------------------------------------------------ #

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._image_bgr is None:
            return

        # 中ボタン or パンモード → パン
        if event.button() == Qt.MouseButton.MiddleButton or self._edit_mode == EditMode.PAN:
            if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton):
                self._pan_active = True
                self._pan_last = event.position().toPoint()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if event.button() not in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            return

        mode = self._edit_mode

        if mode == EditMode.BRUSH:
            self._brush_add = (event.button() == Qt.MouseButton.LeftButton)
            if self._editor is not None:
                self._editor.begin_stroke()
                self._stroke_started = True
            self._painting = True
            self._do_paint(event.position())

        elif mode in (EditMode.RECT_ADD, EditMode.RECT_DEL):
            if event.button() == Qt.MouseButton.LeftButton:
                mx, my = self._widget_to_mask(event.position())
                if mx is not None:
                    self._rect_start = (mx, my)
                    self._rect_end = (mx, my)
                    self._rect_dragging = True

        elif mode in (EditMode.POLY_ADD, EditMode.POLY_DEL):
            if event.button() == Qt.MouseButton.LeftButton:
                mx, my = self._widget_to_mask(event.position())
                if mx is not None:
                    self._poly_points.append((mx, my))
                    self.update()
            elif event.button() == Qt.MouseButton.RightButton:
                self._poly_points.clear()
                self.update()

        elif mode == EditMode.AI_PROMPT:
            mx, my = self._widget_to_mask(event.position())
            if mx is None:
                return
            if event.button() == Qt.MouseButton.LeftButton:
                # 左: クリック=正点 / ドラッグ=矩形
                self._ai_box_start = (mx, my)
                self._ai_box_end = (mx, my)
                self._ai_box_dragging = False
                self._ai_press_widget_pos = event.position()
            elif event.button() == Qt.MouseButton.RightButton:
                # 右: 負点
                self.ai_point_clicked.emit({"x": mx, "y": my, "positive": False})

        elif mode in _GRABCUT_MODES:
            if event.button() == Qt.MouseButton.LeftButton:
                mx, my = self._widget_to_mask(event.position())
                if mx is None:
                    return

                # ヒントモードが有効な場合はヒント描画
                if (self._gc_hint_is_active and
                        self._gc_ui_state in (GrabCutUiState.HINT_EDITING, GrabCutUiState.PREVIEW)):
                    if self._gc_ui_state == GrabCutUiState.PREVIEW:
                        self._set_gc_ui_state(GrabCutUiState.HINT_EDITING)
                    self._gc_hint_drawing = True
                    self._gc_current_stroke_pts = [(mx, my)]
                    self._gc_hint_redo_stack.clear()  # 新しい描画でRedoスタックをクリア

                elif self._gc_ui_state in (GrabCutUiState.IDLE, GrabCutUiState.PREVIEW):
                    # 新しいGrabCut矩形を開始
                    if self._gc_ui_state == GrabCutUiState.PREVIEW:
                        # 既存プレビューをキャンセルして新しい矩形を開始
                        self._cancel_session_internal()
                    self._rect_start = (mx, my)
                    self._rect_end = (mx, my)
                    self._rect_dragging = True

            elif event.button() == Qt.MouseButton.RightButton:
                # ヒント描画中なら現在のストロークをキャンセル
                if self._gc_hint_drawing:
                    self._gc_hint_drawing = False
                    self._gc_current_stroke_pts.clear()
                    self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._image_bgr is None:
            return

        pos = event.position()

        if self._pan_active and self._pan_last:
            delta = pos.toPoint() - self._pan_last
            self._offset += QPointF(delta)
            self._pan_last = pos.toPoint()
            self.update()
            return

        mx, my = self._widget_to_mask(pos)
        self._cursor_pos = (mx, my) if mx is not None else None

        if self._painting and self._edit_mode == EditMode.BRUSH:
            self._do_paint(pos)
        elif self._edit_mode == EditMode.AI_PROMPT and self._ai_box_start is not None \
                and event.buttons() & Qt.MouseButton.LeftButton:
            # 一定距離動いたら矩形ドラッグとみなす
            if self._ai_press_widget_pos is not None:
                d = (pos - self._ai_press_widget_pos)
                if abs(d.x()) + abs(d.y()) > 5:
                    self._ai_box_dragging = True
            if mx is not None:
                self._ai_box_end = (mx, my)
            self.update()
        elif self._gc_hint_drawing and mx is not None:
            # ヒントストローク描画中
            self._gc_current_stroke_pts.append((mx, my))
            self.update()
        elif self._rect_dragging and self._rect_start is not None:
            if mx is not None:
                self._rect_end = (mx, my)
            self.update()
        else:
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton or (
            self._edit_mode == EditMode.PAN and event.button() == Qt.MouseButton.LeftButton
        ):
            self._pan_active = False
            self._pan_last = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return

        if self._edit_mode == EditMode.BRUSH:
            self._painting = False
            self._stroke_started = False

        elif self._edit_mode == EditMode.AI_PROMPT:
            if event.button() == Qt.MouseButton.LeftButton and self._ai_box_start is not None:
                if self._ai_box_dragging and self._ai_box_end is not None:
                    x0, y0 = self._ai_box_start
                    x1, y1 = self._ai_box_end
                    if abs(x1 - x0) >= 4 and abs(y1 - y0) >= 4:
                        self.ai_box_drawn.emit({
                            "x1": min(x0, x1), "y1": min(y0, y1),
                            "x2": max(x0, x1), "y2": max(y0, y1),
                        })
                else:
                    # クリック = 正点
                    x0, y0 = self._ai_box_start
                    self.ai_point_clicked.emit({"x": x0, "y": y0, "positive": True})
                self._ai_box_start = None
                self._ai_box_end = None
                self._ai_box_dragging = False
                self._ai_press_widget_pos = None
                self.update()

        elif self._edit_mode in (EditMode.RECT_ADD, EditMode.RECT_DEL):
            if self._rect_dragging and self._rect_start and self._rect_end:
                self._apply_rect()
            self._rect_start = None
            self._rect_end = None
            self._rect_dragging = False
            self.update()

        elif self._edit_mode in _GRABCUT_MODES:
            if self._gc_hint_drawing:
                # ヒントストロークを確定
                if self._gc_current_stroke_pts:
                    from core.grabcut_tool import HintStroke
                    stroke = HintStroke(
                        label=self._gc_hint_label,
                        points=list(self._gc_current_stroke_pts),
                        radius=self._gc_hint_radius,
                    )
                    self._gc_hint_strokes.append(stroke)
                self._gc_hint_drawing = False
                self._gc_current_stroke_pts.clear()
                self.update()

            elif self._rect_dragging and self._rect_start and self._rect_end:
                self._request_grabcut_preview()
            self._rect_start = None
            self._rect_end = None
            self._rect_dragging = False
            self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else (1.0 / 1.15)
        cursor_pos = event.position()

        old_scale = self._scale
        self._scale = max(0.05, min(50.0, self._scale * factor))

        ratio = self._scale / old_scale
        self._offset = QPointF(
            cursor_pos.x() - ratio * (cursor_pos.x() - self._offset.x()),
            cursor_pos.y() - ratio * (cursor_pos.y() - self._offset.y()),
        )
        self.update()

    # ------------------------------------------------------------------ #
    # キーボード
    # ------------------------------------------------------------------ #

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        mode = self._edit_mode
        ctrl = event.modifiers() & Qt.KeyboardModifier.ControlModifier
        shift = event.modifiers() & Qt.KeyboardModifier.ShiftModifier

        if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.set_brush_radius(self._brush_radius + 5)
            return
        if key == Qt.Key.Key_Minus:
            self.set_brush_radius(self._brush_radius - 5)
            return

        # GrabCutセッションが有効な場合の特殊キー処理
        if self._gc_ui_state != GrabCutUiState.IDLE:
            # Ctrl+Enter → 再推定
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and ctrl:
                if self._gc_ui_state in (GrabCutUiState.PREVIEW, GrabCutUiState.HINT_EDITING):
                    self.request_grabcut_refine()
                return

            # Ctrl+Shift+Z → ヒント全消去
            if key == Qt.Key.Key_Z and ctrl and shift:
                if self._gc_ui_state == GrabCutUiState.HINT_EDITING:
                    self.gc_clear_hints()
                return

            # Ctrl+Z → ヒントUndo (HINT_EDITING時のみ)
            if key == Qt.Key.Key_Z and ctrl and not shift:
                if self._gc_ui_state == GrabCutUiState.HINT_EDITING:
                    self.gc_undo_hint()
                    return
                # IDLE/PREVIEW ではフォールスルーして通常Undoへ

            # Ctrl+Y → ヒントRedo (HINT_EDITING時のみ)
            if key == Qt.Key.Key_Y and ctrl:
                if self._gc_ui_state == GrabCutUiState.HINT_EDITING:
                    self.gc_redo_hint()
                    return

            # Enter → 適用 (処理中でない場合)
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not ctrl:
                if self._gc_ui_state in (GrabCutUiState.PREVIEW, GrabCutUiState.HINT_EDITING):
                    self.apply_grabcut_preview()
                return

            # Esc → キャンセル
            if key == Qt.Key.Key_Escape:
                if self._gc_ui_state in (GrabCutUiState.INITIAL_RUNNING, GrabCutUiState.REFINE_RUNNING):
                    self.grabcut_cancel_requested.emit()
                elif self._gc_ui_state in (GrabCutUiState.PREVIEW, GrabCutUiState.HINT_EDITING):
                    self._cancel_session_internal()
                    self.grabcut_session_cancelled.emit()
                return

        # GrabCutモード (V0.4A互換 - セッションなし時)
        if mode in _GRABCUT_MODES and self._gc_ui_state == GrabCutUiState.IDLE:
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                # プレビューがある場合は適用 (IDLEではないはずだが念のため)
                if self._grabcut_preview_mask is not None:
                    self.apply_grabcut_preview()
                return
            if key == Qt.Key.Key_Escape:
                if self.grabcut_processing:
                    self.grabcut_cancel_requested.emit()
                else:
                    self.cancel_grabcut_preview()
                return

        if mode in (EditMode.POLY_ADD, EditMode.POLY_DEL):
            if key == Qt.Key.Key_Return or key == Qt.Key.Key_Enter:
                self._confirm_polygon()
                return
            if key == Qt.Key.Key_Escape:
                self._poly_points.clear()
                self.update()
                return
            if key == Qt.Key.Key_Backspace:
                if self._poly_points:
                    self._poly_points.pop()
                    self.update()
                return

        super().keyPressEvent(event)

    # ------------------------------------------------------------------ #
    # 矩形・ポリゴン適用
    # ------------------------------------------------------------------ #

    def _apply_rect(self) -> None:
        if self._editor is None or self._rect_start is None or self._rect_end is None:
            return
        x0, y0 = self._rect_start
        x1, y1 = self._rect_end
        lx, rx = min(x0, x1), max(x0, x1)
        ty, by = min(y0, y1), max(y0, y1)
        if lx == rx or ty == by:
            return

        self._editor.begin_stroke()
        color = 255 if self._edit_mode == EditMode.RECT_ADD else 0
        cv2.rectangle(self._editor.mask, (lx, ty), (rx, by), color, -1)
        self.mask_changed.emit()
        self.update()

    def _confirm_polygon(self) -> None:
        if self._editor is None or len(self._poly_points) < 3:
            self._poly_points.clear()
            self.update()
            return

        pts = np.array(self._poly_points, dtype=np.int32).reshape(-1, 1, 2)
        self._editor.begin_stroke()
        color = 255 if self._edit_mode == EditMode.POLY_ADD else 0
        cv2.fillPoly(self._editor.mask, [pts], color)
        self._poly_points.clear()
        self.mask_changed.emit()
        self.update()

    # ------------------------------------------------------------------ #
    # GrabCut処理 (シグナル委譲)
    # ------------------------------------------------------------------ #

    def _request_grabcut_preview(self) -> None:
        """
        矩形ドラッグ後にGrabCutリクエストをシグナルでメインウィンドウへ送る。
        実際の処理はワーカースレッドで行われる。
        """
        if self._editor is None or self._image_bgr is None:
            return
        if self._rect_start is None or self._rect_end is None:
            return

        if self.grabcut_processing:
            self.status_message.emit("GrabCut処理中です。完了をお待ちください。", 3000)
            return

        x0, y0 = self._rect_start
        x1, y1 = self._rect_end
        lx, rx = min(x0, x1), max(x0, x1)
        ty, by = min(y0, y1), max(y0, y1)

        ih, iw = self._image_bgr.shape[:2]
        lx = max(0, lx)
        ty = max(0, ty)
        rx = min(rx, iw)
        by = min(by, ih)

        rw, rh = rx - lx, by - ty
        if rw < 4 or rh < 4:
            self.status_message.emit("矩形が小さすぎます。広めに指定してください。", 4000)
            return

        mode_map = {
            EditMode.GRABCUT_ADD:     "add",
            EditMode.GRABCUT_DEL:     "remove",
            EditMode.GRABCUT_REPLACE: "replace",
        }
        mode_str = mode_map[self._edit_mode]

        from core.grabcut_tool import GrabCutOptions
        options = GrabCutOptions(
            iter_count=self._grabcut_iter_count,
            use_downscale=self._grabcut_use_downscale,
            max_processing_size=self._grabcut_max_processing_size,
            use_existing_mask_as_bgd=self._grabcut_use_existing_mask_as_bgd,
        )

        self._set_gc_ui_state(GrabCutUiState.INITIAL_RUNNING)
        self._grabcut_preview_mode = mode_str
        self._grabcut_rect = (lx, ty, rw, rh)

        # 既存マスクを背景制約として使用する場合はコピーを渡す
        current_mask_copy = None
        if self._grabcut_use_existing_mask_as_bgd and self._editor is not None:
            current_mask_copy = self._editor.mask.copy()

        self.grabcut_requested.emit({
            "image": self._image_bgr.copy(),
            "rect": (lx, ty, rw, rh),
            "mode": mode_str,
            "options": options,
            "current_mask": current_mask_copy,
        })
        self.status_message.emit("GrabCut処理中...", 0)
        self.update()

    def apply_grabcut_preview(self) -> None:
        """
        プレビュー中のGrabCut結果を現在マスクへ適用する。
        適用前に begin_stroke() を呼び、Undo可能にする。
        """
        if self._grabcut_preview_mask is None or self._editor is None:
            return
        if self._grabcut_preview_mode is None:
            return

        from core.grabcut_tool import apply_grabcut_result

        gc_mask = self._grabcut_preview_mask.copy()

        # 後処理 (膨張・収縮)
        ks = self._grabcut_post_kernel_size
        if self._grabcut_post_dilate:
            from core.mask_morphology import dilate_mask
            gc_mask = dilate_mask(gc_mask, ks)
        if self._grabcut_post_erode:
            from core.mask_morphology import erode_mask
            gc_mask = erode_mask(gc_mask, ks)

        self._editor.begin_stroke()
        new_mask = apply_grabcut_result(
            self._editor.mask, gc_mask, self._grabcut_preview_mode
        )
        self._editor.mask[:] = new_mask

        # セッション全体をクリア
        self._grabcut_preview_mask = None
        self._grabcut_preview_mode = None
        self._grabcut_rect = None
        self._gc_hint_strokes.clear()
        self._gc_hint_redo_stack.clear()
        self._gc_hint_drawing = False
        self._gc_current_stroke_pts.clear()
        self._gc_hint_is_active = False
        self._set_gc_ui_state(GrabCutUiState.IDLE)

        self.mask_changed.emit()
        self.status_message.emit("GrabCutを適用しました", 3000)
        self.update()

    def cancel_grabcut_preview(self) -> None:
        """GrabCutプレビューを破棄する。Undo履歴は増やさない。"""
        if self._grabcut_preview_mask is None:
            return
        self._cancel_session_internal()
        self.grabcut_session_cancelled.emit()
        self.status_message.emit("GrabCutをキャンセルしました", 2000)

    def _cancel_session_internal(self) -> None:
        """GrabCutセッション内部状態をクリア (通常マスクは変更しない)。"""
        self._grabcut_preview_mask = None
        self._grabcut_preview_mode = None
        self._grabcut_rect = None
        self._gc_hint_strokes.clear()
        self._gc_hint_redo_stack.clear()
        self._gc_hint_drawing = False
        self._gc_current_stroke_pts.clear()
        self._gc_hint_is_active = False
        self._set_gc_ui_state(GrabCutUiState.IDLE)
        self.update()

    # ------------------------------------------------------------------ #
    # 内部ヘルパー
    # ------------------------------------------------------------------ #

    def _set_gc_ui_state(self, state: GrabCutUiState) -> None:
        """GrabCut UI状態を更新しシグナルを発火する。"""
        if self._gc_ui_state != state:
            self._gc_ui_state = state
            self.grabcut_state_changed.emit(state)

    def _do_paint(self, pos: QPointF) -> None:
        if self._editor is None:
            return
        mx, my = self._widget_to_mask(pos)
        if mx is None:
            return
        self._editor.paint(mx, my, self._brush_radius, self._brush_add)
        self.mask_changed.emit()
        self.update()

    def _widget_to_mask(self, pos: QPointF) -> tuple[Optional[int], Optional[int]]:
        if self._image_bgr is None or self._scale == 0:
            return None, None
        x = (pos.x() - self._offset.x()) / self._scale
        y = (pos.y() - self._offset.y()) / self._scale
        h, w = self._image_bgr.shape[:2]
        mx, my = int(round(x)), int(round(y))
        if 0 <= mx < w and 0 <= my < h:
            return mx, my
        return None, None

    def _mask_to_widget(self, mx: int, my: int) -> tuple[int, int]:
        wx = int(mx * self._scale + self._offset.x())
        wy = int(my * self._scale + self._offset.y())
        return wx, wy

    def _fit_to_view(self) -> None:
        if self._image_bgr is None:
            return
        h, w = self._image_bgr.shape[:2]
        vw, vh = self.width(), self.height()
        if vw <= 0 or vh <= 0:
            self._scale = 1.0
            self._offset = QPointF(0, 0)
            return
        scale = min(vw / w, vh / h) * 0.95
        self._scale = scale
        self._offset = QPointF(
            (vw - w * scale) / 2,
            (vh - h * scale) / 2,
        )

    def _reset_edit_state(self) -> None:
        self._poly_points.clear()
        self._rect_start = None
        self._rect_end = None
        self._rect_dragging = False
        self._painting = False
        self._stroke_started = False
        # GrabCutセッション全体をクリア
        self._grabcut_preview_mask = None
        self._grabcut_preview_mode = None
        self._grabcut_rect = None
        self._gc_hint_strokes.clear()
        self._gc_hint_redo_stack.clear()
        self._gc_hint_drawing = False
        self._gc_current_stroke_pts.clear()
        self._gc_hint_is_active = False
        # AIプロンプト表示状態 (画像切替時にクリア。セッション側もreset)
        self._ai_points = []
        self._ai_box = None
        self._ai_preview_mask = None
        self._ai_box_start = None
        self._ai_box_end = None
        self._ai_box_dragging = False
        # _gc_ui_state はワーカーライフサイクルで管理するためここでは触らない
        # (Workerが実行中に画像が切り替わった場合はMainWindowがclear_grabcut_stateを呼ぶ)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        if self._image_bgr is not None:
            self._fit_to_view()
        super().resizeEvent(event)


def _bgr_to_qimage(bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
