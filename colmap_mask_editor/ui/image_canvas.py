"""
中央キャンバス: 画像表示・マスク半透明重ね・ブラシ編集・ズーム・パン
"""

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


class ImageCanvas(QWidget):
    """
    画像とマスクを重ね表示し、ブラシ編集・ズーム・パンを提供するウィジェット。
    """

    mask_changed = Signal()  # ブラシ操作でマスクが変化したとき

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

        # 表示対象
        self._image_bgr: Optional[np.ndarray] = None   # BGR画像
        self._editor: Optional[MaskEditor] = None

        # 表示設定
        self._mask_opacity: float = 0.45       # マスク透明度 0.0-1.0
        self._mask_visible: bool = True
        self._mask_color: tuple[int, int, int] = (255, 60, 60)  # RGB

        # ブラシ設定
        self._brush_radius: int = 20
        self._brush_add: bool = True           # True=追加, False=削除
        self._painting: bool = False
        self._stroke_started: bool = False

        # ズーム・パン
        self._scale: float = 1.0
        self._offset: QPointF = QPointF(0, 0)  # キャンバス上の画像左上位置
        self._pan_last: Optional[QPoint] = None
        self._pan_active: bool = False

        # カーソル位置(マスク座標)
        self._cursor_pos: Optional[tuple[int, int]] = None

        self.setMinimumSize(400, 300)

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #

    def set_image(self, image_bgr: np.ndarray) -> None:
        """表示画像をセット。マスクエディタはリセット"""
        self._image_bgr = image_bgr
        self._editor = None
        self._fit_to_view()
        self.update()

    def set_editor(self, editor: MaskEditor) -> None:
        """マスクエディタをセット"""
        self._editor = editor
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

    def clear(self) -> None:
        self._image_bgr = None
        self._editor = None
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

        # 合成画像を生成してQPixmapに変換
        composite = self._build_composite()
        qimg = _bgr_to_qimage(composite)
        pixmap = QPixmap.fromImage(qimg)

        # スケール・オフセット適用
        painter.save()
        painter.translate(self._offset)
        w = int(self._image_bgr.shape[1] * self._scale)
        h = int(self._image_bgr.shape[0] * self._scale)
        painter.drawPixmap(QRect(0, 0, w, h), pixmap)
        painter.restore()

        # ブラシカーソル描画
        if self._cursor_pos:
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

    def _build_composite(self) -> np.ndarray:
        """画像とマスクを合成してBGR配列を返す"""
        img = self._image_bgr.copy()
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if self._mask_visible and self._editor is not None:
            mask = self._editor.mask
            if mask.shape[:2] == img.shape[:2]:
                overlay = img.copy()
                r, g, b = self._mask_color
                overlay[mask == 255] = [b, g, r]  # BGR
                alpha = self._mask_opacity
                img = cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0)

        return img

    # ------------------------------------------------------------------ #
    # マウス操作
    # ------------------------------------------------------------------ #

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._image_bgr is None:
            return

        if event.button() == Qt.MouseButton.MiddleButton:
            # パン開始
            self._pan_active = True
            self._pan_last = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if event.button() == Qt.MouseButton.LeftButton:
            self._brush_add = True
        elif event.button() == Qt.MouseButton.RightButton:
            self._brush_add = False
        else:
            return

        if self._editor is not None:
            self._editor.begin_stroke()
            self._stroke_started = True
        self._painting = True
        self._do_paint(event.position())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._image_bgr is None:
            return

        pos = event.position()

        # パン処理
        if self._pan_active and self._pan_last:
            delta = pos.toPoint() - self._pan_last
            self._offset += QPointF(delta)
            self._pan_last = pos.toPoint()
            self.update()
            return

        # マスク座標更新
        mx, my = self._widget_to_mask(pos)
        if mx is not None:
            self._cursor_pos = (mx, my)
        else:
            self._cursor_pos = None

        if self._painting:
            self._do_paint(pos)
        else:
            self.update()  # カーソルだけ再描画

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_active = False
            self._pan_last = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return

        self._painting = False
        self._stroke_started = False

    def wheelEvent(self, event: QWheelEvent) -> None:
        """ホイールでズーム(カーソル位置を中心に)"""
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else (1.0 / 1.15)
        cursor_pos = event.position()

        old_scale = self._scale
        self._scale = max(0.05, min(50.0, self._scale * factor))

        # カーソル位置を中心にオフセット調整
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
        if key == Qt.Key.Key_Plus or key == Qt.Key.Key_Equal:
            self.set_brush_radius(self._brush_radius + 5)
        elif key == Qt.Key.Key_Minus:
            self.set_brush_radius(self._brush_radius - 5)
        else:
            super().keyPressEvent(event)

    # ------------------------------------------------------------------ #
    # 内部ヘルパー
    # ------------------------------------------------------------------ #

    def _do_paint(self, pos: QPointF) -> None:
        """ウィジェット座標からマスク座標に変換してブラシ描画"""
        if self._editor is None:
            return
        mx, my = self._widget_to_mask(pos)
        if mx is None:
            return
        self._editor.paint(mx, my, self._brush_radius, self._brush_add)
        self.mask_changed.emit()
        self.update()

    def _widget_to_mask(self, pos: QPointF) -> tuple[Optional[int], Optional[int]]:
        """ウィジェット座標をマスク(画像)座標に変換"""
        if self._image_bgr is None or self._scale == 0:
            return None, None
        x = (pos.x() - self._offset.x()) / self._scale
        y = (pos.y() - self._offset.y()) / self._scale
        h, w = self._image_bgr.shape[:2]
        mx, my = int(round(x)), int(round(y))
        if 0 <= mx < w and 0 <= my < h:
            return mx, my
        return None, None

    def _fit_to_view(self) -> None:
        """画像をビューに収まるようスケールとオフセットを設定"""
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

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        if self._image_bgr is not None:
            self._fit_to_view()
        super().resizeEvent(event)


def _bgr_to_qimage(bgr: np.ndarray) -> QImage:
    """BGR numpy配列を QImage(RGB888)に変換"""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
