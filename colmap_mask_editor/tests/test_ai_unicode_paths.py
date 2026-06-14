"""
日本語・全角スペースを含む Windows パスの取り扱いテスト。

  - プロトコルでの往復 (test_ai_protocol にもあるが、ここでは実ファイルで検証)
  - 画像ローダ (worker側) が日本語/全角スペースパスを読める
  - NPZ 一時ファイルがそのようなランタイムディレクトリ下で読み書きできる
"""

import numpy as np
import cv2
import pytest

from ai.ai_mask_ops import load_prediction_npz
from sam_backend.image_loader import load_image_bgr, load_image_rgb
from sam_backend.result_writer import write_result_npz


@pytest.fixture
def jp_dir(tmp_path):
    d = tmp_path / "プロジェクト 全角　スペース" / "画像"
    d.mkdir(parents=True)
    return d


def test_load_image_jp_path(jp_dir):
    img = np.zeros((20, 30, 3), dtype=np.uint8)
    img[:, :, 2] = 255  # 赤 (BGR)
    path = jp_dir / "画像　001.jpg"  # 全角スペース入り
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    buf.tofile(str(path))

    bgr = load_image_bgr(str(path))
    assert bgr.shape == (20, 30, 3)

    rgb, w, h = load_image_rgb(str(path))
    assert (w, h) == (30, 20)
    assert rgb.shape == (20, 30, 3)


def test_load_grayscale_becomes_bgr(jp_dir):
    gray = np.full((10, 12), 128, dtype=np.uint8)
    path = jp_dir / "グレー　画像.png"
    ok, buf = cv2.imencode(".png", gray)
    assert ok
    buf.tofile(str(path))
    bgr = load_image_bgr(str(path))
    assert bgr.ndim == 3 and bgr.shape[2] == 3


def test_npz_roundtrip_in_jp_runtime_dir(tmp_path, monkeypatch):
    rt = tmp_path / "ランタイム　dir"
    monkeypatch.setenv("COLMAP_MASK_EDITOR_RUNTIME_DIR", str(rt))
    masks = np.zeros((2, 8, 8), dtype=np.uint8)
    masks[0, 1:4, 1:4] = 255
    scores = np.array([0.9, 0.5], dtype=np.float32)
    path = write_result_npz(masks, scores, request_id=3, image_key="キー")
    result = load_prediction_npz(path, expected_request_id=3, expected_image_key="キー")
    assert result.mask_count == 2
    assert result.image_key == "キー"
