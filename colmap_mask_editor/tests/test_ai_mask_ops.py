"""
AI結果NPZの読み込み・候補統計・通常マスクへの適用テスト。
"""

import numpy as np
import pytest

from ai.ai_mask_ops import (
    APPLY_ADD,
    APPLY_EXCLUDE,
    APPLY_REPLACE,
    NpzCorruptError,
    apply_ai_mask,
    load_prediction_npz,
)
from sam_backend.result_writer import write_result_npz


@pytest.fixture
def runtime_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("COLMAP_MASK_EDITOR_RUNTIME_DIR", str(tmp_path / "rt"))
    return tmp_path / "rt"


def _make_masks(h=40, w=50):
    masks = np.zeros((3, h, w), dtype=np.uint8)
    masks[0, 5:35, 5:45] = 255
    masks[1, 10:30, 10:40] = 255
    masks[2, 0:40, 0:50] = 255  # 全面
    return masks


def test_write_and_load_roundtrip(runtime_dir):
    masks = _make_masks()
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    path = write_result_npz(masks, scores, request_id=42, image_key="key1")

    result = load_prediction_npz(path, expected_request_id=42, expected_image_key="key1")
    assert result.mask_count == 3
    assert result.request_id == 42
    assert result.image_key == "key1"
    assert result.width == 50 and result.height == 40
    assert result.best_index() == 0  # 最大スコア


def test_candidate_stats(runtime_dir):
    masks = _make_masks()
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    path = write_result_npz(masks, scores, 1, "k")
    result = load_prediction_npz(path)
    c2 = result.candidates[2]  # 全面マスク
    assert c2.fg_pixels == 40 * 50
    assert c2.fg_ratio == pytest.approx(1.0)
    assert c2.size == (50, 40)


def test_load_rejects_wrong_request_id(runtime_dir):
    masks = _make_masks()
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    path = write_result_npz(masks, scores, 5, "k")
    with pytest.raises(NpzCorruptError):
        load_prediction_npz(path, expected_request_id=999)


def test_load_rejects_wrong_image_key(runtime_dir):
    masks = _make_masks()
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    path = write_result_npz(masks, scores, 5, "key_a")
    with pytest.raises(NpzCorruptError):
        load_prediction_npz(path, expected_image_key="key_b")


def test_load_corrupt_npz(tmp_path):
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"not a real npz")
    with pytest.raises(NpzCorruptError):
        load_prediction_npz(bad)


def test_load_missing_file(tmp_path):
    with pytest.raises(NpzCorruptError):
        load_prediction_npz(tmp_path / "nope.npz")


# ----- 適用 (追加/除外/置換) -----

def test_apply_add():
    current = np.zeros((10, 10), dtype=np.uint8)
    ai = np.zeros((10, 10), dtype=np.uint8)
    ai[2:5, 2:5] = 255
    out = apply_ai_mask(current, ai, APPLY_ADD)
    assert out[3, 3] == 255
    assert out[0, 0] == 0
    # 元配列は変更されない
    assert current[3, 3] == 0


def test_apply_exclude():
    current = np.full((10, 10), 255, dtype=np.uint8)
    ai = np.zeros((10, 10), dtype=np.uint8)
    ai[2:5, 2:5] = 255
    out = apply_ai_mask(current, ai, APPLY_EXCLUDE)
    assert out[3, 3] == 0
    assert out[0, 0] == 255


def test_apply_replace():
    current = np.full((10, 10), 255, dtype=np.uint8)
    ai = np.zeros((10, 10), dtype=np.uint8)
    ai[2:5, 2:5] = 255
    out = apply_ai_mask(current, ai, APPLY_REPLACE)
    assert out[3, 3] == 255
    assert out[0, 0] == 0
    assert set(np.unique(out)).issubset({0, 255})


def test_apply_size_mismatch_raises():
    current = np.zeros((10, 10), dtype=np.uint8)
    ai = np.zeros((5, 5), dtype=np.uint8)
    with pytest.raises(ValueError):
        apply_ai_mask(current, ai, APPLY_ADD)


def test_apply_invalid_mode():
    current = np.zeros((4, 4), dtype=np.uint8)
    ai = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        apply_ai_mask(current, ai, "frobnicate")


def test_atomic_write_no_tmp_left(runtime_dir):
    masks = _make_masks()
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    write_result_npz(masks, scores, 99, "k")
    # .tmp_ ファイルが残っていない
    leftovers = list(runtime_dir.glob(".tmp_*"))
    assert leftovers == []
