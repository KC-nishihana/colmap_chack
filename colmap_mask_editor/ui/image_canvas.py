"""
中央キャンバス: 画像表示・マスク半透明重ね・ブラシ/矩形/ポリゴン/GrabCut編集・差分表示・ズーム・パン
v0.4A.1: GrabCut処理をシグナルで委譲し、処理中フラグで二重起動を防止
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
    PAN = auto()


_MODE_LABEL = {
    EditMode.BRUSH:          "ブラシ",
    EditMode.RECT_ADD:       "矩形追加",
    EditMode.RECT_DEL:       "矩形削除",
    EditMode.POLY_ADD:       "ポリゴン追加",
    EditMode.POLY_DEL:       "ポリゴン削除",
    EditMode.GRABCUT_ADD:    "GrabCut有効化",
    EditMode.GRABCUT_DEL:    "GrabCut除外",
    EditMode.GRABCUT_REPLACE:"GrabCut置換",
    EditMode.PAN:            "パン操作",
}

_GRABCUT_MODES = (EditMode.GRABCUT_ADD, EditMode.GRABCUT_DEL, EditMode.GRABCUT_REPLACE)


class ImageCanvas(QWidget):
    """
    画像とマスクを重ね表示し、各種編集・ズーム・パンを提供するウィジェット。
    """

    mask_changed = Signal()
    mode_changed = Signal(str)           # 編集モード名
    status_message = Signal(str, int)    # メッセージ, タイムアウトms (0=持続)

    # GrabCut処理をメインウィンドウ側へ委譲するシグナル
    # dict: {"image": ndarray, "rect": tuple, "mode": str, "options": GrabCutOptions}
    grabcut_requested = Signal(object)

    # キャンセル要求 (処理中にEscを押した場合)
    grabcut_cancel_requested = Signal()

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

        # GrabCutプレビュー状態
        self._grabcut_preview_mask: Optional[np.ndarray] = None
        self._grabcut_preview_mode: Optional[str] = None
        self._grabcut_rect: Optional[tuple[int, int, int, int]] = None

        # GrabCut処理中フラグ (二重起動防止)
        self._grabcut_processing: bool = False

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

    # ----- GrabCut状態 (読み取り) -----

    @property
    def grabcut_processing(self) -> bool:
        return self._grabcut_processing

    # ----- GrabCutプレビューのセット/クリア (Workerから呼ばれる) -----

    def set_grabcut_preview(self, mask: np.ndarray, mode_str: str) -> None:
        """Worker処理完了後にメインウィンドウから呼ばれてプレビューをセットする。"""
        self._grabcut_preview_mask = mask
        self._grabcut_preview_mode = mode_str
        self._grabcut_processing = False
        self.status_message.emit("GrabCutプレビュー中: Enter=適用 / Esc=キャンセル", 0)
        self.update()

    def clear_grabcut_state(self) -> None:
        """エラー・キャンセル時にプレビューと処理フラグをクリアする。"""
        self._grabcut_preview_mask = None
        self._grabcut_preview_mode = None
        self._grabcut_rect = None
        self._grabcut_processing = False
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

        composite = self._build_composite()
        qimg = _bgr_to_qimage(composite)
        pixmap = QPixmap.fromImage(qimg)

        painter.save()
        painter.translate(self._offset)
        w = int(self._image_bgr.shape[1] * self._scale)
        h = int(self._image_bgr.shape[0] * self._scale)
        painter.drawPixmap(QRect(0, 0, w, h), pixmap)
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
        if self._grabcut_processing:
            painter.save()
            painter.setPen(QColor(255, 200, 0))
            painter.drawText(10, 20, "GrabCut処理中...")
            painter.restore()

    def _build_composite(self) -> np.ndarray:
        img = self._image_bgr.copy()
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        result = img
        if self._mask_visible and self._editor is not None:
            mask = self._editor.mask
            if self._diff_mode and self._baseline_mask is not None and mask.shape == self._baseline_mask.shape:
                result = self._build_diff_composite(img, mask)
            elif mask.shape[:2] == img.shape[:2]:
                overlay = img.copy()
                r, g, b = self._mask_color
                overlay[mask == 255] = [b, g, r]  # BGR
                result = cv2.addWeighted(overlay, self._mask_opacity, img, 1.0 - self._mask_opacity, 0)

        # GrabCutプレビューをマスク表示ON/OFFに関わらず重ねる
        if (self._grabcut_preview_mask is not None
                and self._grabcut_preview_mask.shape[:2] == img.shape[:2]):
            result = self._overlay_grabcut_preview(result)

        return result

    def _build_diff_composite(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """差分表示: 追加=緑, 削除=青, 変化なし=赤半透明"""
        base = self._baseline_mask
        result = img.copy()

        added   = (mask == 255) & (base == 0)    # 追加領域: 緑
        removed = (mask == 0)   & (base == 255)  # 削除領域: 青
        kept    = (mask == 255) & (base == 255)  # 変化なし: 赤半透明

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

    def _overlay_grabcut_preview(self, base: np.ndarray) -> np.ndarray:
        """GrabCutプレビュー結果を半透明オーバーレイで表示"""
        if self._grabcut_preview_mask is None:
            return base

        region = self._grabcut_preview_mask == 255
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

        elif mode in _GRABCUT_MODES:
            if event.button() == Qt.MouseButton.LeftButton:
                mx, my = self._widget_to_mask(event.position())
                if mx is not None:
                    self._rect_start = (mx, my)
                    self._rect_end = (mx, my)
                    self._rect_dragging = True

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

        elif self._edit_mode in (EditMode.RECT_ADD, EditMode.RECT_DEL):
            if self._rect_dragging and self._rect_start and self._rect_end:
                self._apply_rect()
            self._rect_start = None
            self._rect_end = None
            self._rect_dragging = False
            self.update()

        elif self._edit_mode in _GRABCUT_MODES:
            if self._rect_dragging and self._rect_start and self._rect_end:
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

        if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.set_brush_radius(self._brush_radius + 5)
            return
        if key == Qt.Key.Key_Minus:
            self.set_brush_radius(self._brush_radius - 5)
            return

        # GrabCutモード中はEnter/EscをGrabCut処理に使う (ポリゴンより優先)
        if mode in _GRABCUT_MODES:
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if not self._grabcut_processing:
                    self.apply_grabcut_preview()
                return
            if key == Qt.Key.Key_Escape:
                if self._grabcut_processing:
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

        if self._grabcut_processing:
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
        )

        self._grabcut_processing = True
        self._grabcut_preview_mode = mode_str
        self._grabcut_rect = (lx, ty, rw, rh)

        self.grabcut_requested.emit({
            "image": self._image_bgr.copy(),
            "rect": (lx, ty, rw, rh),
            "mode": mode_str,
            "options": options,
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

        self._grabcut_preview_mask = None
        self._grabcut_preview_mode = None
        self._grabcut_rect = None

        self.mask_changed.emit()
        self.status_message.emit("GrabCutを適用しました", 3000)
        self.update()

    def cancel_grabcut_preview(self) -> None:
        """GrabCutプレビューを破棄する。Undo履歴は増やさない。"""
        if self._grabcut_preview_mask is None:
            return
        self._grabcut_preview_mask = None
        self._grabcut_preview_mode = None
        self._grabcut_rect = None
        self.status_message.emit("GrabCutをキャンセルしました", 2000)
        self.update()

    # ------------------------------------------------------------------ #
    # 内部ヘルパー
    # ------------------------------------------------------------------ #

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
        self._grabcut_preview_mask = None
        self._grabcut_preview_mode = None
        self._grabcut_rect = None
        # _grabcut_processing はワーカーライフサイクルで管理するためここでは触らない

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        if self._image_bgr is not None:
            self._fit_to_view()
        super().resizeEvent(event)


def _bgr_to_qimage(bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
