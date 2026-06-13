"""
大画像処理テスト: ROI切り出しと縮小・復元処理の検証 (v0.4A.1)

巨大配列でのメモリ消費を避けるため、テスト画像は 2400x1600 を使用する。
GrabCutアルゴリズムの精度ではなく、座標変換・縮小・復元処理を検証する。
"""

import cv2
import numpy as np
import pytest

from core.grabcut_tool import (
    GrabCutOptions,
    GrabCutResult,
    _compute_roi,
    _validate_options,
    _validate_rect,
    run_grabcut_optimized,
)


# ------------------------------------------------------------------ #
# テスト用ヘルパー
# ------------------------------------------------------------------ #

IMG_W, IMG_H = 2400, 1600


def make_large_test_image() -> np.ndarray:
    """2400x1600 のテスト用BGR画像"""
    image = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    image[:] = (30, 30, 30)
    # 中央に明るい矩形を配置
    image[400:1200, 600:1800] = (200, 200, 200)
    return image


def make_dummy_grabcut(monkeypatch):
    """
    cv2.grabCut を差し替えて、GCマスクの中央部分を GC_PR_FGD にするダミー実装。
    これにより座標変換・縮小・復元処理のみを検証できる。
    """
    original_grabcut = cv2.grabCut

    def dummy_grabcut(img, mask, rect, bgd_model, fgd_model, iter_count, mode):
        h, w = mask.shape
        # 処理サイズの中央領域を前景候補に設定
        cx, cy = w // 2, h // 2
        hw, hh = w // 4, h // 4
        mask[max(0, cy - hh):cy + hh, max(0, cx - hw):cx + hw] = cv2.GC_PR_FGD

    monkeypatch.setattr(cv2, "grabCut", dummy_grabcut)
    return original_grabcut


# ------------------------------------------------------------------ #
# ROI・縮小計算の検証 (GrabCutを実行しない)
# ------------------------------------------------------------------ #

class TestROIComputation:
    def test_roi_includes_margin_around_rect(self):
        rect = (600, 400, 600, 400)
        opts = _validate_options(GrabCutOptions(min_margin_px=32, roi_margin_ratio=0.10))
        roi_x, roi_y, roi_w, roi_h = _compute_roi(rect, IMG_W, IMG_H, opts)

        # ROIは矩形より大きい
        assert roi_x <= 600
        assert roi_y <= 400
        assert roi_x + roi_w >= 600 + 600
        assert roi_y + roi_h >= 400 + 400

    def test_roi_clipped_to_image_boundary(self):
        # 画像端の矩形
        rect = (0, 0, 100, 100)
        opts = _validate_options(GrabCutOptions(min_margin_px=50, roi_margin_ratio=0.10))
        roi_x, roi_y, roi_w, roi_h = _compute_roi(rect, IMG_W, IMG_H, opts)

        assert roi_x >= 0
        assert roi_y >= 0
        assert roi_x + roi_w <= IMG_W
        assert roi_y + roi_h <= IMG_H

    def test_scale_respects_max_processing_size(self):
        max_size = 1024
        roi_w, roi_h = 2000, 1500
        scale = min(1.0, max_size / max(roi_w, roi_h))
        proc_w = int(roi_w * scale)
        proc_h = int(roi_h * scale)

        assert max(proc_w, proc_h) <= max_size
        assert scale < 1.0


# ------------------------------------------------------------------ #
# run_grabcut_optimized 大画像統合テスト
# ------------------------------------------------------------------ #

class TestLargeImageProcessing:

    def test_downscaled_when_max_size_1024(self, monkeypatch):
        make_dummy_grabcut(monkeypatch)
        image = make_large_test_image()
        # rect=(600,400,600,400) の ROI は約720x480 < 1024 → 縮小されない。
        # ROI が 1024px を超えるよう幅広の矩形を使用する (ROI ≈ 1440x960)。
        rect = (400, 200, 1200, 800)
        opts = GrabCutOptions(iter_count=1, use_downscale=True, max_processing_size=1024)

        result = run_grabcut_optimized(image, rect, opts)

        assert result.was_downscaled is True

    def test_processing_size_long_side_at_most_1024(self, monkeypatch):
        make_dummy_grabcut(monkeypatch)
        image = make_large_test_image()
        rect = (600, 400, 600, 400)
        opts = GrabCutOptions(iter_count=1, use_downscale=True, max_processing_size=1024)

        result = run_grabcut_optimized(image, rect, opts)

        pw, ph = result.processing_size
        assert max(pw, ph) <= 1024

    def test_output_mask_matches_original_image_size(self, monkeypatch):
        make_dummy_grabcut(monkeypatch)
        image = make_large_test_image()
        rect = (600, 400, 600, 400)
        opts = GrabCutOptions(iter_count=1, use_downscale=True, max_processing_size=1024)

        result = run_grabcut_optimized(image, rect, opts)

        assert result.mask.shape == (IMG_H, IMG_W)

    def test_output_mask_dtype_uint8(self, monkeypatch):
        make_dummy_grabcut(monkeypatch)
        image = make_large_test_image()
        rect = (600, 400, 600, 400)
        opts = GrabCutOptions(iter_count=1, use_downscale=True, max_processing_size=1024)

        result = run_grabcut_optimized(image, rect, opts)

        assert result.mask.dtype == np.uint8

    def test_output_mask_only_0_and_255(self, monkeypatch):
        make_dummy_grabcut(monkeypatch)
        image = make_large_test_image()
        rect = (600, 400, 600, 400)
        opts = GrabCutOptions(iter_count=1, use_downscale=True, max_processing_size=1024)

        result = run_grabcut_optimized(image, rect, opts)

        unique = set(np.unique(result.mask).tolist())
        assert unique.issubset({0, 255})

    def test_roi_outside_area_is_zero(self, monkeypatch):
        """ROI外のマスクは必ず0"""
        make_dummy_grabcut(monkeypatch)
        image = make_large_test_image()
        rect = (600, 400, 600, 400)
        opts = GrabCutOptions(iter_count=1, use_downscale=True, max_processing_size=1024,
                              min_margin_px=32, roi_margin_ratio=0.10)

        result = run_grabcut_optimized(image, rect, opts)

        roi_x, roi_y, roi_w, roi_h = result.roi
        # ROI外の角を確認
        if roi_x > 5:
            assert result.mask[0, 0] == 0
        if roi_y > 5:
            assert result.mask[0, 0] == 0

    def test_original_image_not_modified(self, monkeypatch):
        """元画像配列が変更されていないことを確認"""
        make_dummy_grabcut(monkeypatch)
        image = make_large_test_image()
        original_copy = image.copy()
        rect = (600, 400, 600, 400)
        opts = GrabCutOptions(iter_count=1, use_downscale=True, max_processing_size=1024)

        run_grabcut_optimized(image, rect, opts)

        assert np.array_equal(image, original_copy)

    def test_inter_nearest_preserves_binary_values(self, monkeypatch):
        """INTER_NEARESTで復元後も0/255のみであることを確認"""
        make_dummy_grabcut(monkeypatch)
        image = make_large_test_image()
        rect = (500, 300, 800, 600)
        opts = GrabCutOptions(iter_count=1, use_downscale=True, max_processing_size=512)

        result = run_grabcut_optimized(image, rect, opts)

        assert result.was_downscaled is True
        unique = set(np.unique(result.mask).tolist())
        assert unique.issubset({0, 255}), f"中間値が含まれています: {unique}"

    def test_no_downscale_when_disabled(self, monkeypatch):
        make_dummy_grabcut(monkeypatch)
        image = make_large_test_image()
        rect = (600, 400, 200, 150)
        opts = GrabCutOptions(iter_count=1, use_downscale=False)

        result = run_grabcut_optimized(image, rect, opts)

        assert result.was_downscaled is False
        assert result.scale == pytest.approx(1.0)
