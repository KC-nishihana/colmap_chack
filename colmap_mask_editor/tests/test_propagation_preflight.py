"""伝播事前検証のテスト (torch不要)。"""

import numpy as np

from ai.propagation_preflight import DimEntry, validate_reference_mask, validate_sequence


def _good_mask(w=64, h=48):
    m = np.zeros((h, w), np.uint8)
    m[10:20, 10:20] = 255
    return m


def test_reference_mask_ok():
    assert validate_reference_mask(_good_mask(), 64, 48) == []


def test_reference_mask_size_mismatch():
    errs = validate_reference_mask(_good_mask(64, 48), 32, 32)
    assert any("サイズ" in e for e in errs)


def test_reference_mask_not_uint8():
    m = _good_mask().astype(np.int32)
    errs = validate_reference_mask(m, 64, 48)
    assert any("uint8" in e for e in errs)


def test_reference_mask_not_binary():
    m = _good_mask()
    m[0, 0] = 128
    errs = validate_reference_mask(m, 64, 48)
    assert any("0/255" in e for e in errs)


def test_reference_mask_empty():
    m = np.zeros((48, 64), np.uint8)
    errs = validate_reference_mask(m, 64, 48)
    assert any("前景が0" in e for e in errs)


def test_reference_mask_full():
    m = np.full((48, 64), 255, np.uint8)
    errs = validate_reference_mask(m, 64, 48)
    assert any("全体" in e for e in errs)


def _dims(n, w=100, h=100):
    return [DimEntry(f"e{i}", f"{i}.jpg", w, h) for i in range(n)]


def test_sequence_ok():
    assert validate_sequence(_dims(5), "e2", max_frames=100) == []


def test_sequence_too_few():
    errs = validate_sequence(_dims(1), "e0", max_frames=100)
    assert any("2枚未満" in e for e in errs)


def test_sequence_size_mismatch():
    dims = _dims(3)
    dims[1] = DimEntry("e1", "e1.jpg", 200, 100)
    errs = validate_sequence(dims, "e0", max_frames=100)
    assert any("異なる画像サイズ" in e for e in errs)


def test_sequence_reference_missing():
    errs = validate_sequence(_dims(3), "zzz", max_frames=100)
    assert any("基準画像が対象範囲内に存在しません" in e for e in errs)


def test_sequence_duplicate():
    dims = _dims(3)
    dims.append(DimEntry("e0", "e0.jpg", 100, 100))
    errs = validate_sequence(dims, "e0", max_frames=100)
    assert any("重複" in e for e in errs)


def test_sequence_exceeds_max():
    errs = validate_sequence(_dims(5), "e0", max_frames=3)
    assert any("最大値" in e for e in errs)
