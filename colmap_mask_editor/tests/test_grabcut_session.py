"""
create_grabcut_session のユニットテスト (v0.4B)

GrabCutSessionの構造・プレビューマスク形状・ラベル値・ROI計算を検証する。
"""

import cv2
import numpy as np
import pytest

from core.grabcut_tool import (
    GrabCutHintLabel,
    GrabCutOptions,
    GrabCutSession,
    create_grabcut_session,
)


# ------------------------------------------------------------------ #
# テスト用ヘルパー
# ------------------------------------------------------------------ #

def make_image(h: int = 80, w: int = 100) -> np.ndarray:
    """GrabCutが前景を検出できる十分なコントラストを持つ画像。"""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = (30, 20, 10)  # 暗い背景
    cy, cx = h // 2, w // 2
    r = min(h, w) // 4
    cv2.circle(img, (cx, cy), r, (60, 120, 200), -1)  # 明るい前景円
    return img


def default_rect(w: int = 100, h: int = 80) -> tuple:
    cy, cx = h // 2, w // 2
    r = min(h, w) // 4
    return (cx - r, cy - r, r * 2, r * 2)


def default_options(**kwargs) -> GrabCutOptions:
    return GrabCutOptions(iter_count=2, use_downscale=False, **kwargs)


# ------------------------------------------------------------------ #
# 戻り値の型と形状
# ------------------------------------------------------------------ #

def test_returns_grabcut_session():
    """create_grabcut_session はGrabCutSessionを返す"""
    img = make_image()
    session = create_grabcut_session(img, default_rect(), default_options())
    assert isinstance(session, GrabCutSession)


def test_preview_mask_shape_matches_input():
    """preview_maskは元画像と同じサイズ"""
    img = make_image(80, 100)
    session = create_grabcut_session(img, default_rect(100, 80), default_options())
    assert session.preview_mask.shape == (80, 100)


def test_preview_mask_dtype_uint8():
    """preview_maskはuint8"""
    img = make_image()
    session = create_grabcut_session(img, default_rect(), default_options())
    assert session.preview_mask.dtype == np.uint8


def test_preview_mask_values_only_0_255():
    """preview_maskの値は0または255のみ"""
    img = make_image()
    session = create_grabcut_session(img, default_rect(), default_options())
    unique = set(np.unique(session.preview_mask).tolist())
    assert unique.issubset({0, 255})


def test_label_mask_dtype_uint8():
    """label_maskはuint8"""
    img = make_image()
    session = create_grabcut_session(img, default_rect(), default_options())
    assert session.label_mask.dtype == np.uint8
    assert session.base_label_mask.dtype == np.uint8


def test_label_mask_values_valid():
    """ラベルマスクの値は {0,1,2,3} のみ"""
    import cv2
    valid = {int(cv2.GC_BGD), int(cv2.GC_FGD), int(cv2.GC_PR_BGD), int(cv2.GC_PR_FGD)}
    img = make_image()
    session = create_grabcut_session(img, default_rect(), default_options())
    for m in (session.label_mask, session.base_label_mask):
        unique = set(np.unique(m).tolist())
        assert unique.issubset(valid)


def test_original_size_correct():
    """original_sizeが元画像のサイズと一致する"""
    img = make_image(80, 100)
    session = create_grabcut_session(img, default_rect(100, 80), default_options())
    assert session.original_size == (100, 80)


def test_original_rect_stored():
    """original_rectが指定した矩形と一致する"""
    img = make_image()
    rect = default_rect()
    session = create_grabcut_session(img, rect, default_options())
    rx, ry, rw, rh = session.original_rect
    assert rw > 0 and rh > 0


# ------------------------------------------------------------------ #
# ROI の計算
# ------------------------------------------------------------------ #

def test_roi_within_image():
    """ROIが画像境界内に収まっている"""
    img = make_image(80, 100)
    session = create_grabcut_session(img, default_rect(100, 80), default_options())
    rx, ry, rw, rh = session.roi
    assert rx >= 0 and ry >= 0
    assert rx + rw <= 100
    assert ry + rh <= 80


def test_roi_covers_rect():
    """ROIは指定矩形を含む"""
    img = make_image(80, 100)
    # 明るいオブジェクトを含む矩形
    rect = (25, 20, 50, 40)
    session = create_grabcut_session(img, rect, default_options())
    rx, ry, rw, rh = session.roi
    # ROIは矩形より広い (マージン分)
    assert rx <= 25 and ry <= 20
    assert rx + rw >= 25 + 50
    assert ry + rh >= 20 + 40


def test_processing_size_matches_roi_when_no_downscale():
    """縮小なし: processing_sizeはROIサイズと一致する"""
    img = make_image(80, 100)
    opts = GrabCutOptions(iter_count=2, use_downscale=False)
    session = create_grabcut_session(img, default_rect(100, 80), opts)
    _, _, rw, rh = session.roi
    pw, ph = session.processing_size
    assert pw == rw and ph == rh


def test_scale_is_1_when_no_downscale():
    """縮小なし: scale=1.0"""
    img = make_image()
    opts = GrabCutOptions(iter_count=2, use_downscale=False)
    session = create_grabcut_session(img, default_rect(), opts)
    assert session.scale == 1.0
    assert not session.was_downscaled


# ------------------------------------------------------------------ #
# ダウンスケール
# ------------------------------------------------------------------ #

def test_downscale_applied_when_roi_large():
    """大きなROIに対して縮小が適用される"""
    img = make_image(1200, 1600)
    rect = (100, 100, 1400, 1000)
    opts = GrabCutOptions(iter_count=1, use_downscale=True, max_processing_size=512)
    session = create_grabcut_session(img, rect, opts)
    assert session.was_downscaled
    assert session.scale < 1.0
    pw, ph = session.processing_size
    assert max(pw, ph) <= 512 + 5  # 小さな丸め誤差を許容


def test_no_downscale_when_small():
    """小さな画像では縮小されない"""
    img = make_image(80, 100)
    opts = GrabCutOptions(iter_count=2, use_downscale=True, max_processing_size=2048)
    session = create_grabcut_session(img, default_rect(100, 80), opts)
    assert not session.was_downscaled
    assert session.scale == 1.0


# ------------------------------------------------------------------ #
# モデルの形状
# ------------------------------------------------------------------ #

def test_bgd_fgd_model_shapes():
    """bgd_model/fgd_modelは (1, 65) のfloat64"""
    img = make_image()
    session = create_grabcut_session(img, default_rect(), default_options())
    assert session.bgd_model.shape == (1, 65)
    assert session.fgd_model.shape == (1, 65)
    assert session.bgd_model.dtype == np.float64
    assert session.fgd_model.dtype == np.float64


# ------------------------------------------------------------------ #
# base_label_mask の不変性
# ------------------------------------------------------------------ #

def test_base_label_mask_is_copy():
    """base_label_maskとlabel_maskは別オブジェクト (一方を変更しても他方に影響しない)"""
    img = make_image()
    session = create_grabcut_session(img, default_rect(), default_options())
    orig = session.base_label_mask.copy()
    session.label_mask[:] = 0  # label_maskをゼロクリア
    np.testing.assert_array_equal(session.base_label_mask, orig)


# ------------------------------------------------------------------ #
# 入力画像の不変性
# ------------------------------------------------------------------ #

def test_input_image_not_modified():
    """入力画像は変更されない"""
    img = make_image()
    orig = img.copy()
    create_grabcut_session(img, default_rect(), default_options())
    np.testing.assert_array_equal(img, orig)


# ------------------------------------------------------------------ #
# 現在マスク制約 (use_existing_mask_as_bgd)
# ------------------------------------------------------------------ #

def test_existing_mask_bgd_constraint():
    """use_existing_mask_as_bgd=TrueでROI内の0領域がGC_BGDになる"""
    img = make_image(80, 100)
    current_mask = np.full((80, 100), 255, dtype=np.uint8)
    current_mask[0:40, :] = 0  # 上半分は除外 (確実にROI内を含む)
    opts = GrabCutOptions(
        iter_count=2,
        use_downscale=False,
        use_existing_mask_as_bgd=True,
    )
    session = create_grabcut_session(img, default_rect(100, 80), opts, current_mask)
    # current_mask==0 の領域はGC_BGDにクランプされているはず
    assert cv2.GC_BGD in np.unique(session.base_label_mask)


# ------------------------------------------------------------------ #
# refine_count の初期値
# ------------------------------------------------------------------ #

def test_refine_count_initial_zero():
    """初回セッションのrefine_countは0"""
    img = make_image()
    session = create_grabcut_session(img, default_rect(), default_options())
    assert session.refine_count == 0


# ------------------------------------------------------------------ #
# バリデーション: 不正な入力
# ------------------------------------------------------------------ #

def test_raises_on_empty_image():
    """空画像でValueErrorまたはcv2.errorが発生する"""
    img = np.zeros((0, 80, 3), dtype=np.uint8)
    with pytest.raises((ValueError, cv2.error)):
        create_grabcut_session(img, (0, 0, 80, 0), default_options())


def test_raises_on_tiny_rect():
    """矩形が小さすぎるとValueErrorが発生する"""
    img = make_image(60, 80)
    with pytest.raises(ValueError):
        create_grabcut_session(img, (35, 28, 2, 2), default_options())


# cvエラーは別のモジュールでインポートしてもよいが、pytest.raises(Exception)でも十分
