"""
apply_hint_strokes のユニットテスト (v0.4B)

ヒント座標変換・FG/BG/消去の動作・ROI境界外の無視・エッジケースを検証する。
"""

import cv2
import numpy as np
import pytest

from core.grabcut_tool import (
    GrabCutHintLabel,
    GrabCutOptions,
    GrabCutSession,
    HintStroke,
    apply_hint_strokes,
    create_grabcut_session,
)


# ------------------------------------------------------------------ #
# テスト用ヘルパー
# ------------------------------------------------------------------ #

def make_image(h: int = 80, w: int = 100) -> np.ndarray:
    """GrabCutが前景を検出できる十分なコントラストを持つ画像。"""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # 外縁: 暗い青背景
    img[:, :] = (40, 30, 20)
    # 中央領域: 明るい赤 (前景)
    cy, cx = h // 2, w // 2
    r = min(h, w) // 4
    cv2.circle(img, (cx, cy), r, (60, 120, 200), -1)
    return img


def make_session(h: int = 80, w: int = 100) -> GrabCutSession:
    """縮小なしでシンプルなセッションを作成する。"""
    img = make_image(h, w)
    cy, cx = h // 2, w // 2
    r = min(h, w) // 4
    rect = (cx - r, cy - r, r * 2, r * 2)
    opts = GrabCutOptions(iter_count=2, use_downscale=False)
    return create_grabcut_session(img, rect, opts)


# ------------------------------------------------------------------ #
# 基本: FG ヒント
# ------------------------------------------------------------------ #

def test_fg_hint_sets_gc_fgd():
    """FGヒントのピクセルがGC_FGDに設定される"""
    session = make_session()
    roi_x, roi_y, roi_w, roi_h = session.roi
    cx = roi_x + roi_w // 2
    cy = roi_y + roi_h // 2
    stroke = HintStroke(label=GrabCutHintLabel.FOREGROUND, points=[(cx, cy)], radius=5)

    result = apply_hint_strokes(session, [stroke])

    # ストロークの中心近くがGC_FGD (=1) になっているはず
    px = int(round((cx - roi_x) * session.scale))
    py = int(round((cy - roi_y) * session.scale))
    assert result[py, px] == cv2.GC_FGD


def test_bg_hint_sets_gc_bgd():
    """BGヒントのピクセルがGC_BGDに設定される"""
    session = make_session()
    roi_x, roi_y, roi_w, roi_h = session.roi
    cx = roi_x + roi_w // 2
    cy = roi_y + roi_h // 2
    stroke = HintStroke(label=GrabCutHintLabel.BACKGROUND, points=[(cx, cy)], radius=5)

    result = apply_hint_strokes(session, [stroke])

    px = int(round((cx - roi_x) * session.scale))
    py = int(round((cy - roi_y) * session.scale))
    assert result[py, px] == cv2.GC_BGD


# ------------------------------------------------------------------ #
# 消去ヒント
# ------------------------------------------------------------------ #

def test_erase_hint_restores_base_label_mask():
    """消去ヒントがbase_label_maskの値に戻る"""
    session = make_session()
    roi_x, roi_y, roi_w, roi_h = session.roi
    cx = roi_x + roi_w // 2
    cy = roi_y + roi_h // 2

    # まずFGヒントを適用
    fg_stroke = HintStroke(label=GrabCutHintLabel.FOREGROUND, points=[(cx, cy)], radius=5)
    result_fg = apply_hint_strokes(session, [fg_stroke])

    # 中心がFGDになっていることを確認
    px = int(round((cx - roi_x) * session.scale))
    py = int(round((cy - roi_y) * session.scale))
    assert result_fg[py, px] == cv2.GC_FGD

    # 次に同位置に消去ヒントを追加
    erase_stroke = HintStroke(label=None, points=[(cx, cy)], radius=5)
    result_erased = apply_hint_strokes(session, [fg_stroke, erase_stroke])

    # base_label_maskの値に戻っているはず
    assert result_erased[py, px] == session.base_label_mask[py, px]


def test_erase_does_not_use_pr_bgd():
    """消去ヒントはGC_PR_BGD(2)を設定しない (base_label_maskへ戻すのみ)"""
    session = make_session()
    roi_x, roi_y, roi_w, roi_h = session.roi
    cx = roi_x + roi_w // 2
    cy = roi_y + roi_h // 2
    erase_stroke = HintStroke(label=None, points=[(cx, cy)], radius=5)

    result = apply_hint_strokes(session, [erase_stroke])

    # 消去後の値がbase_label_maskと一致すること
    px = int(round((cx - roi_x) * session.scale))
    py = int(round((cy - roi_y) * session.scale))
    assert result[py, px] == session.base_label_mask[py, px]


# ------------------------------------------------------------------ #
# ROI境界外の点の無視
# ------------------------------------------------------------------ #

def test_roi_outside_points_ignored():
    """ROI境界外の点は無視される (クラッシュしない)"""
    session = make_session(60, 80)
    roi_x, roi_y, roi_w, roi_h = session.roi

    # ROI外の座標
    outside_pts = [
        (0, 0),             # 左上端
        (79, 59),           # 右下端
        (roi_x - 5, roi_y), # ROI左外
        (roi_x, roi_y - 5), # ROI上外
        (roi_x + roi_w + 5, roi_y),  # ROI右外
    ]
    stroke = HintStroke(label=GrabCutHintLabel.FOREGROUND, points=outside_pts, radius=3)

    # クラッシュしないこと
    result = apply_hint_strokes(session, [stroke])
    assert result.shape == session.label_mask.shape


def test_empty_strokes_returns_copy_of_base():
    """ストロークなしの場合はbase_label_maskのコピーが返る"""
    session = make_session()
    result = apply_hint_strokes(session, [])
    np.testing.assert_array_equal(result, session.base_label_mask)


def test_empty_stroke_points_ignored():
    """pointsが空のストロークは無視される"""
    session = make_session()
    stroke = HintStroke(label=GrabCutHintLabel.FOREGROUND, points=[], radius=5)
    result = apply_hint_strokes(session, [stroke])
    np.testing.assert_array_equal(result, session.base_label_mask)


# ------------------------------------------------------------------ #
# ストローク連続点の接続 (ラインで補間)
# ------------------------------------------------------------------ #

def test_consecutive_points_connected():
    """連続した2点がラインで補間されギャップが生じない"""
    session = make_session(60, 80)
    roi_x, roi_y, roi_w, roi_h = session.roi
    # ROI内の2点 (充分離れている)
    p1x = roi_x + 5
    p1y = roi_y + 5
    p2x = roi_x + roi_w - 5
    p2y = roi_y + roi_h - 5
    stroke = HintStroke(
        label=GrabCutHintLabel.FOREGROUND,
        points=[(p1x, p1y), (p2x, p2y)],
        radius=3,
    )
    result = apply_hint_strokes(session, [stroke])

    # 2点間の中間付近もGC_FGDになっていること
    mx = (p1x + p2x) // 2
    my = (p1y + p2y) // 2
    px = int(round((mx - roi_x) * session.scale))
    py = int(round((my - roi_y) * session.scale))
    ph, pw = result.shape
    px = max(0, min(px, pw - 1))
    py = max(0, min(py, ph - 1))
    assert result[py, px] == cv2.GC_FGD


# ------------------------------------------------------------------ #
# 画像端での大きなブラシ
# ------------------------------------------------------------------ #

def test_large_brush_at_image_edge_no_crash():
    """画像端での大きなブラシサイズでクラッシュしない"""
    session = make_session(60, 80)
    roi_x, roi_y = session.roi[:2]
    # ROI左上端 (ギリギリROI内)
    stroke = HintStroke(
        label=GrabCutHintLabel.BACKGROUND,
        points=[(roi_x, roi_y)],
        radius=200,  # 非常に大きい
    )
    result = apply_hint_strokes(session, [stroke])
    assert result.shape == session.label_mask.shape


# ------------------------------------------------------------------ #
# ヒント適用後のラベル値の検証
# ------------------------------------------------------------------ #

def test_result_values_are_valid_gc_labels():
    """apply_hint_strokes後のラベル値は有効なGrabCutラベル (0-3) のみ"""
    session = make_session()
    roi_x, roi_y, roi_w, roi_h = session.roi
    cx = roi_x + roi_w // 2
    cy = roi_y + roi_h // 2
    strokes = [
        HintStroke(label=GrabCutHintLabel.FOREGROUND, points=[(cx, cy)], radius=5),
        HintStroke(label=GrabCutHintLabel.BACKGROUND, points=[(roi_x + 2, roi_y + 2)], radius=3),
    ]
    result = apply_hint_strokes(session, strokes)

    valid = {int(cv2.GC_BGD), int(cv2.GC_FGD), int(cv2.GC_PR_BGD), int(cv2.GC_PR_FGD)}
    unique = set(np.unique(result).tolist())
    assert unique.issubset(valid)


# ------------------------------------------------------------------ #
# 入力Sessionの不変性
# ------------------------------------------------------------------ #

def test_session_not_modified_by_apply_hints():
    """apply_hint_strokes は入力Sessionを破壊しない"""
    session = make_session()
    base_copy = session.base_label_mask.copy()
    label_copy = session.label_mask.copy()

    roi_x, roi_y, roi_w, roi_h = session.roi
    cx = roi_x + roi_w // 2
    cy = roi_y + roi_h // 2
    stroke = HintStroke(label=GrabCutHintLabel.FOREGROUND, points=[(cx, cy)], radius=10)

    apply_hint_strokes(session, [stroke])

    np.testing.assert_array_equal(session.base_label_mask, base_copy)
    np.testing.assert_array_equal(session.label_mask, label_copy)


# ------------------------------------------------------------------ #
# 座標変換の検証 (スケール付き)
# ------------------------------------------------------------------ #

def test_coordinate_transform_with_scale():
    """縮小時に元画像座標が正しく処理座標へ変換される"""
    img = make_image(1200, 1600)
    rect = (100, 100, 1400, 1000)
    opts = GrabCutOptions(iter_count=1, use_downscale=True, max_processing_size=512)
    session = create_grabcut_session(img, rect, opts)

    assert session.was_downscaled  # 縮小が適用されていること

    roi_x, roi_y, roi_w, roi_h = session.roi
    scale = session.scale

    # ROI内の点
    orig_x = roi_x + roi_w // 2
    orig_y = roi_y + roi_h // 2

    stroke = HintStroke(
        label=GrabCutHintLabel.FOREGROUND,
        points=[(orig_x, orig_y)],
        radius=5,
    )
    result = apply_hint_strokes(session, [stroke])

    # 期待する処理座標
    expected_px = int(round((orig_x - roi_x) * scale))
    expected_py = int(round((orig_y - roi_y) * scale))
    ph, pw = result.shape
    expected_px = max(0, min(expected_px, pw - 1))
    expected_py = max(0, min(expected_py, ph - 1))

    assert result[expected_py, expected_px] == cv2.GC_FGD
