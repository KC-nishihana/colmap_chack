"""
V0.9: 完全被覆リージョンの階層レビュー画面 (PySide6)。

中央キャンバスに元画像・リージョン境界・KEEP/REMOVE/未確認オーバーレイを描画し、
クリックで判断、ダブルクリックで局所細分化、Backspace で親へ戻す。表示は
PartitionReviewModel を介して操作し、判断は別途 partition_review.json へ保存する。

torch / sam2 を import しない (numpy + OpenCV + PySide6 のみ)。重い分割・統合処理は
このウィジェットでは行わない (生成は CPU Worker、ここでは表示と判断のみ)。
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy

from ai.partition_review_model import PartitionReviewModel
from ai.partition_review_state import RegionDecision

# 表示用ラベルマップの長辺上限 (8K でも軽量に描画する)
_DISPLAY_MAX_SIDE = 1280

_COLORS = {
    "keep": (0, 200, 0),
    "remove": (220, 30, 30),
    "unreviewed": (150, 150, 150),
}


class PartitionReviewWidget(QLabel):
    decision_changed = Signal()
    selection_changed = Signal(int)
    visible_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._model: PartitionReviewModel | None = None
        self._disp_labels: np.ndarray | None = None   # 表示解像度の葉ラベル
        self._disp_scale = 1.0
        self._img_h = self._img_w = 0
        self._selected: int | None = None
        self._show_boundaries = True
        self._overlay_alpha = 0.5
        self._render_rect = (0, 0, 1, 1)  # pixmap 描画先 (x,y,w,h)

    # ---- セットアップ ---- #
    def set_partition(self, image_bgr: np.ndarray, model: PartitionReviewModel) -> None:
        self._model = model
        self._img_h, self._img_w = image_bgr.shape[:2]
        scale = min(1.0, _DISPLAY_MAX_SIDE / max(self._img_w, self._img_h))
        dw = max(1, int(round(self._img_w * scale)))
        dh = max(1, int(round(self._img_h * scale)))
        self._disp_scale = scale
        self._disp_img = cv2.resize(image_bgr, (dw, dh), interpolation=cv2.INTER_AREA)
        # 表示解像度の葉ラベル (run-length を復号し最近傍縮小)
        from ai import partition_rle
        full = partition_rle.decode_to_label_map(
            model.data["run_region_ids"], model.data["run_lengths"],
            self._img_h, self._img_w)
        self._disp_labels = cv2.resize(
            full.astype(np.int32), (dw, dh), interpolation=cv2.INTER_NEAREST)
        self._selected = None
        self._render()

    def set_show_boundaries(self, on: bool) -> None:
        self._show_boundaries = bool(on)
        self._render()

    def set_overlay_alpha(self, alpha: float) -> None:
        self._overlay_alpha = max(0.0, min(1.0, float(alpha)))
        self._render()

    # ---- レンダリング ---- #
    def _render(self) -> None:
        if self._model is None or self._disp_labels is None:
            return
        model = self._model
        labels = self._disp_labels
        k = model.tree.leaf_count
        visible_set = set(model.visible)

        # leaf -> 実効判断色 / 表示ノード id の LUT
        dec_color = np.zeros((k + 1, 3), dtype=np.uint8)
        vis_lut = np.zeros(k + 1, dtype=np.int64)
        from ai.partition_tree import leaf_to_visible_node
        parent = model.parent
        for leaf in range(1, k + 1):
            vnode = leaf_to_visible_node(leaf, visible_set, parent)
            vis_lut[leaf] = vnode
            eff = model.effective(vnode)
            dec_color[leaf] = _COLORS.get(eff, _COLORS["unreviewed"])

        base = cv2.cvtColor(self._disp_img, cv2.COLOR_BGR2RGB).astype(np.float32)
        overlay = dec_color[labels].astype(np.float32)
        blended = (base * (1.0 - self._overlay_alpha)
                   + overlay * self._overlay_alpha)

        if self._show_boundaries:
            vis_map = vis_lut[labels]
            b = np.zeros(labels.shape, dtype=bool)
            b[:, :-1] |= vis_map[:, :-1] != vis_map[:, 1:]
            b[:, 1:] |= vis_map[:, :-1] != vis_map[:, 1:]
            b[:-1, :] |= vis_map[:-1, :] != vis_map[1:, :]
            b[1:, :] |= vis_map[:-1, :] != vis_map[1:, :]
            blended[b] = (20, 20, 20)

        # 選択ノードを強調 (白縁)
        if self._selected is not None and self._selected in visible_set:
            sel = vis_lut[labels] == self._selected
            blended[sel] = blended[sel] * 0.6 + np.array([255, 255, 0]) * 0.4

        rgb = np.clip(blended, 0, 255).astype(np.uint8)
        h, w = rgb.shape[:2]
        img = QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(img)
        scaled = pix.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation)
        # 描画先矩形を記録 (クリック逆変換用)
        ox = (self.width() - scaled.width()) // 2
        oy = (self.height() - scaled.height()) // 2
        self._render_rect = (ox, oy, scaled.width(), scaled.height())
        self.setPixmap(scaled)

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._render()

    # ---- 座標変換 ---- #
    def _widget_to_image(self, wx: int, wy: int) -> tuple[int, int] | None:
        ox, oy, rw, rh = self._render_rect
        if rw <= 0 or rh <= 0:
            return None
        if wx < ox or wy < oy or wx >= ox + rw or wy >= oy + rh:
            return None
        fx = (wx - ox) / rw
        fy = (wy - oy) / rh
        ix = int(fx * self._img_w)
        iy = int(fy * self._img_h)
        return max(0, min(self._img_w - 1, ix)), max(0, min(self._img_h - 1, iy))

    # ---- マウス ---- #
    def mousePressEvent(self, event):  # noqa: N802
        if self._model is None:
            return
        pos = self._widget_to_image(int(event.position().x()), int(event.position().y()))
        if pos is None:
            return
        from ai.partition_hit_test import PartitionHitTester
        node = self._hit().node_at(pos[0], pos[1], set(self._model.visible))
        if node is None:
            return
        self._selected = node
        self.selection_changed.emit(node)
        btn = event.button()
        mods = event.modifiers()
        if btn == Qt.MouseButton.LeftButton and (mods & Qt.KeyboardModifier.ControlModifier):
            self._set_decision(node, RegionDecision.UNREVIEWED)
        elif btn == Qt.MouseButton.LeftButton:
            self._set_decision(node, RegionDecision.KEEP)
        elif btn == Qt.MouseButton.RightButton:
            self._set_decision(node, RegionDecision.REMOVE)
        self._render()

    def mouseDoubleClickEvent(self, event):  # noqa: N802
        if self._model is None:
            return
        pos = self._widget_to_image(int(event.position().x()), int(event.position().y()))
        if pos is None:
            return
        node = self._hit().node_at(pos[0], pos[1], set(self._model.visible))
        if node is not None and self._model.split(node):
            self.visible_changed.emit()
            self._render()

    def keyPressEvent(self, event):  # noqa: N802
        if self._model is None or self._selected is None:
            return
        key = event.key()
        if key == Qt.Key.Key_Backspace:
            parent = self._model.collapse(self._selected)
            if parent is not None:
                self._selected = parent
                self.visible_changed.emit()
                self._render()
        elif key == Qt.Key.Key_Space:
            self._show_boundaries = not self._show_boundaries
            self._render()
        elif key == Qt.Key.Key_N:
            forward = not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            nxt = self._model.next_unreviewed(self._selected, forward)
            if nxt is not None:
                self._selected = nxt
                self.selection_changed.emit(nxt)
                self._render()

    # ---- 内部 ---- #
    def _hit(self):
        from ai.partition_hit_test import PartitionHitTester
        if not hasattr(self, "_hit_tester") or self._hit_tester is None:
            self._hit_tester = PartitionHitTester(self._model.data)
        return self._hit_tester

    def _set_decision(self, node, decision):
        clear = False
        if decision != RegionDecision.UNREVIEWED:
            if self._model.descendants_with_decisions(node):
                clear = True  # 簡易: 親設定時は子孫判断を上書き (本来は確認ダイアログ)
        self._model.set_decision(node, decision, clear_descendants=clear)
        self.decision_changed.emit()

    @property
    def selected(self):
        return self._selected
