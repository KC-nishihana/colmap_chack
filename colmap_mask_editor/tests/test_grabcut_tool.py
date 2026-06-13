"""
core/grabcut_tool.py の単体テスト (v0.4A.1)

テスト対象:
  - apply_grabcut_result
  - _validate_and_convert_image
  - _validate_rect
  - run_grabcut_optimized の戻り値検証
"""

import cv2
import numpy as np
import pytest

from core.grabcut_tool import (
    GrabCutOptions,
    _validate_and_convert_image,
    _validate_rect,
    apply_grabcut_result,
    run_grabcut_optimized,
)


def _patch_grabcut_center_foreground(monkeypatch):
    """cv2.grabCut をダミー実装で差し替え: 中央領域を GC_PR_FGD にする"""
    def dummy(img, mask, rect, bgd, fgd, iters, mode):
        h, w = mask.shape
        cx, cy = w // 2, h // 2
        hw, hh = max(1, w // 4), max(1, h // 4)
        mask[max(0, cy - hh):cy + hh, max(0, cx - hw):cx + hw] = cv2.GC_PR_FGD
    monkeypatch.setattr(cv2, "grabCut", dummy)


# ------------------------------------------------------------------ #
# テスト用ヘルパー
# ------------------------------------------------------------------ #

def make_test_image(width: int = 320, height: int = 240) -> np.ndarray:
    """テスト用BGR画像: 暗背景に明るい矩形を持つ小画像"""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (40, 40, 40)
    cv2_available = True
    try:
        import cv2
        cv2.rectangle(image, (80, 50), (240, 190), (220, 220, 220), -1)
    except Exception:
        image[50:190, 80:240] = (220, 220, 220)
    return image


def make_mask(width: int = 320, height: int = 240, value: int = 0) -> np.ndarray:
    return np.full((height, width), value, dtype=np.uint8)


# ------------------------------------------------------------------ #
# apply_grabcut_result
# ------------------------------------------------------------------ #

class TestApplyGrabcutResult:
    def test_add_sets_region_to_255(self):
        current = make_mask(value=0)
        gc_mask = make_mask(value=0)
        gc_mask[50:100, 50:100] = 255

        result = apply_grabcut_result(current, gc_mask, "add")

        assert result[70, 70] == 255
        assert result[10, 10] == 0

    def test_remove_sets_region_to_0(self):
        current = make_mask(value=255)
        gc_mask = make_mask(value=0)
        gc_mask[50:100, 50:100] = 255

        result = apply_grabcut_result(current, gc_mask, "remove")

        assert result[70, 70] == 0
        assert result[10, 10] == 255

    def test_replace_replaces_entire_mask(self):
        current = make_mask(value=255)
        gc_mask = make_mask(value=0)
        gc_mask[0:50, 0:50] = 255

        result = apply_grabcut_result(current, gc_mask, "replace")

        assert np.array_equal(result, gc_mask)

    def test_unknown_mode_raises_value_error(self):
        with pytest.raises(ValueError, match="不明なmode"):
            apply_grabcut_result(make_mask(), make_mask(), "unknown")

    def test_does_not_mutate_input_mask(self):
        current = make_mask(value=0)
        original = current.copy()
        gc_mask = make_mask(value=255)

        apply_grabcut_result(current, gc_mask, "add")

        assert np.array_equal(current, original)


# ------------------------------------------------------------------ #
# _validate_and_convert_image
# ------------------------------------------------------------------ #

class TestValidateAndConvertImage:
    def test_grayscale_converted_to_bgr(self):
        gray = np.zeros((100, 100), dtype=np.uint8)
        result = _validate_and_convert_image(gray)
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_bgra_converted_to_bgr(self):
        bgra = np.zeros((100, 100, 4), dtype=np.uint8)
        result = _validate_and_convert_image(bgra)
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_bgr_shape_preserved(self):
        bgr = np.zeros((100, 200, 3), dtype=np.uint8)
        result = _validate_and_convert_image(bgr)
        assert result.shape == (100, 200, 3)

    def test_invalid_channel_count_raises_value_error(self):
        bad = np.zeros((100, 100, 2), dtype=np.uint8)
        with pytest.raises(ValueError, match="チャンネル数"):
            _validate_and_convert_image(bad)

    def test_none_raises_value_error(self):
        with pytest.raises(ValueError):
            _validate_and_convert_image(None)  # type: ignore

    def test_empty_array_raises_value_error(self):
        with pytest.raises(ValueError):
            _validate_and_convert_image(np.array([]))

    def test_too_small_raises_value_error(self):
        with pytest.raises(ValueError, match="小さすぎます"):
            _validate_and_convert_image(np.zeros((1, 100, 3), dtype=np.uint8))


# ------------------------------------------------------------------ #
# _validate_rect
# ------------------------------------------------------------------ #

class TestValidateRect:
    def test_valid_rect_passes(self):
        result = _validate_rect((10, 10, 100, 80), 320, 240)
        assert result == (10, 10, 100, 80)

    def test_rect_clipped_at_image_boundary(self):
        xc, yc, wc, hc = _validate_rect((300, 220, 100, 100), 320, 240)
        assert xc + wc <= 320
        assert yc + hc <= 240

    def test_negative_x_clipped_to_zero(self):
        xc, yc, wc, hc = _validate_rect((-10, 0, 100, 80), 320, 240)
        assert xc == 0
        assert wc > 0

    def test_zero_width_raises(self):
        with pytest.raises(ValueError):
            _validate_rect((10, 10, 0, 80), 320, 240)

    def test_zero_height_raises(self):
        with pytest.raises(ValueError):
            _validate_rect((10, 10, 100, 0), 320, 240)

    def test_too_small_after_clip_raises(self):
        # x=317, w=10 → clip x2=320 → wc=3 < min_w=5 → raises
        with pytest.raises(ValueError, match="小さすぎます"):
            _validate_rect((317, 10, 10, 80), 320, 240)

    def test_no_intersection_raises(self):
        with pytest.raises(ValueError, match="交差"):
            _validate_rect((400, 0, 100, 100), 320, 240)  # 完全に画像外

    def test_full_image_rect_raises(self):
        with pytest.raises(ValueError, match="近すぎる"):
            _validate_rect((0, 0, 320, 240), 320, 240)

    def test_almost_full_image_rect_raises(self):
        with pytest.raises(ValueError, match="近すぎる"):
            _validate_rect((2, 2, 316, 236), 320, 240)


# ------------------------------------------------------------------ #
# run_grabcut_optimized の戻り値検証
# ------------------------------------------------------------------ #

class TestGrabCutOptimizedReturnValue:
    """run_grabcut_optimized の出力仕様を確認する (GrabCutをmonkeypatching)。"""

    def test_mask_shape_matches_image(self, monkeypatch):
        _patch_grabcut_center_foreground(monkeypatch)
        image = make_test_image(160, 120)
        result = run_grabcut_optimized(image, (40, 25, 80, 70),
                                       GrabCutOptions(iter_count=1, use_downscale=False))
        assert result.mask.shape == (120, 160)

    def test_mask_dtype_is_uint8(self, monkeypatch):
        _patch_grabcut_center_foreground(monkeypatch)
        image = make_test_image(160, 120)
        result = run_grabcut_optimized(image, (40, 25, 80, 70),
                                       GrabCutOptions(iter_count=1, use_downscale=False))
        assert result.mask.dtype == np.uint8

    def test_mask_values_only_0_and_255(self, monkeypatch):
        _patch_grabcut_center_foreground(monkeypatch)
        image = make_test_image(160, 120)
        result = run_grabcut_optimized(image, (40, 25, 80, 70),
                                       GrabCutOptions(iter_count=1, use_downscale=False))
        unique = np.unique(result.mask)
        assert set(unique.tolist()).issubset({0, 255})

    def test_original_size_correct(self, monkeypatch):
        _patch_grabcut_center_foreground(monkeypatch)
        image = make_test_image(160, 120)
        result = run_grabcut_optimized(image, (40, 25, 80, 70),
                                       GrabCutOptions(iter_count=1, use_downscale=False))
        assert result.original_size == (160, 120)

    def test_was_downscaled_false_when_disabled(self, monkeypatch):
        _patch_grabcut_center_foreground(monkeypatch)
        image = make_test_image(160, 120)
        result = run_grabcut_optimized(image, (40, 25, 80, 70),
                                       GrabCutOptions(iter_count=1, use_downscale=False))
        assert result.was_downscaled is False

    def test_scale_is_1_when_not_downscaled(self, monkeypatch):
        _patch_grabcut_center_foreground(monkeypatch)
        image = make_test_image(160, 120)
        result = run_grabcut_optimized(image, (40, 25, 80, 70),
                                       GrabCutOptions(iter_count=1, use_downscale=False))
        assert result.scale == pytest.approx(1.0)


class TestGrabCutOptimizedDownscale:
    """縮小処理の動作検証 (GrabCutをmonkeypatching)。

    _validate_options が max_processing_size を最低256にクランプするため、
    ROIが256pxを超える画像 (400x300) と max_processing_size=256 を使用する。
    rect=(50,40,300,220) の ROI は約364x284 > 256 → scale<1.0 が保証される。
    """
    _IMG_W, _IMG_H = 400, 300
    _RECT = (50, 40, 300, 220)
    _MAX_SIZE = 256

    def test_was_downscaled_true(self, monkeypatch):
        _patch_grabcut_center_foreground(monkeypatch)
        image = make_test_image(self._IMG_W, self._IMG_H)
        result = run_grabcut_optimized(image, self._RECT,
                                       GrabCutOptions(iter_count=1, use_downscale=True,
                                                      max_processing_size=self._MAX_SIZE))
        assert result.was_downscaled is True

    def test_processing_size_within_limit(self, monkeypatch):
        _patch_grabcut_center_foreground(monkeypatch)
        image = make_test_image(self._IMG_W, self._IMG_H)
        result = run_grabcut_optimized(image, self._RECT,
                                       GrabCutOptions(iter_count=1, use_downscale=True,
                                                      max_processing_size=self._MAX_SIZE))
        pw, ph = result.processing_size
        assert max(pw, ph) <= self._MAX_SIZE

    def test_mask_shape_still_matches_original(self, monkeypatch):
        _patch_grabcut_center_foreground(monkeypatch)
        image = make_test_image(self._IMG_W, self._IMG_H)
        result = run_grabcut_optimized(image, self._RECT,
                                       GrabCutOptions(iter_count=1, use_downscale=True,
                                                      max_processing_size=self._MAX_SIZE))
        assert result.mask.shape == (self._IMG_H, self._IMG_W)

    def test_mask_dtype_uint8(self, monkeypatch):
        _patch_grabcut_center_foreground(monkeypatch)
        image = make_test_image(self._IMG_W, self._IMG_H)
        result = run_grabcut_optimized(image, self._RECT,
                                       GrabCutOptions(iter_count=1, use_downscale=True,
                                                      max_processing_size=self._MAX_SIZE))
        assert result.mask.dtype == np.uint8

    def test_mask_values_binary(self, monkeypatch):
        _patch_grabcut_center_foreground(monkeypatch)
        image = make_test_image(self._IMG_W, self._IMG_H)
        result = run_grabcut_optimized(image, self._RECT,
                                       GrabCutOptions(iter_count=1, use_downscale=True,
                                                      max_processing_size=self._MAX_SIZE))
        unique = np.unique(result.mask)
        assert set(unique.tolist()).issubset({0, 255})
