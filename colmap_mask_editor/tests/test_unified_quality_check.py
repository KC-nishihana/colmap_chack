"""V0.11: 保存/確定時の自動品質チェック (unified_quality_check) のテスト。"""

import numpy as np

from ai.unified_quality_check import check_mask_quality


def _mask(h, w, fill=0):
    return np.full((h, w), fill, np.uint8)


def test_good_mask_ok_no_warnings():
    m = _mask(20, 30, 255)
    m[0:10, :] = 0          # 除外率 ~50%
    r = check_mask_quality(m, (20, 30))
    assert r.ok
    assert r.errors == []
    assert r.warnings == []
    assert abs(r.excluded_ratio - 0.5) < 1e-6


def test_size_mismatch_is_error():
    m = _mask(20, 30, 255)
    r = check_mask_quality(m, (21, 30))
    assert not r.ok
    assert any("サイズ" in e for e in r.errors)


def test_non_uint8_is_error():
    m = np.zeros((10, 10), np.float32)
    r = check_mask_quality(m, (10, 10))
    assert not r.ok
    assert any("uint8" in e for e in r.errors)


def test_non_binary_values_is_error():
    m = _mask(10, 10, 255)
    m[0, 0] = 128
    r = check_mask_quality(m, (10, 10))
    assert not r.ok
    assert any("0/255" in e for e in r.errors)


def test_all_zero_warns():
    m = _mask(10, 10, 0)
    r = check_mask_quality(m, (10, 10))
    assert r.ok                      # 致命的ではない
    assert r.all_zero
    assert any("全面0" in w for w in r.warnings)


def test_all_full_warns_zero_exclusion():
    m = _mask(10, 10, 255)
    r = check_mask_quality(m, (10, 10))
    assert r.ok
    assert r.all_full
    assert any("全面255" in w for w in r.warnings)
    # 全面255 のときは「除外率0%」を二重に出さない
    assert not any("除外率が0%" in w for w in r.warnings)


def test_high_exclusion_warns():
    m = _mask(10, 10, 0)
    m[0:1, 0:5] = 255                # 有効 5px / 100 -> 除外率 95%
    r = check_mask_quality(m, (10, 10))
    assert any("除外率が高すぎ" in w for w in r.warnings)


def test_diff_ratio_warns_on_large_change():
    prev = _mask(10, 10, 0)
    cur = _mask(10, 10, 255)         # 全画素反転 -> 差分 100%
    r = check_mask_quality(cur, (10, 10), previous_mask=prev)
    assert r.diff_ratio == 1.0
    assert any("差分が大き" in w for w in r.warnings)


def test_diff_ratio_none_when_no_previous():
    m = _mask(10, 10, 255); m[0:5, :] = 0
    r = check_mask_quality(m, (10, 10))
    assert r.diff_ratio is None


def test_diff_ratio_none_on_shape_mismatch():
    prev = _mask(8, 8, 0)
    cur = _mask(10, 10, 255); cur[0:5, :] = 0
    r = check_mask_quality(cur, (10, 10), previous_mask=prev)
    assert r.diff_ratio is None      # 形不一致は差分計算しない


def test_3d_mask_reduced_to_first_channel():
    m = np.zeros((10, 10, 3), np.uint8); m[..., 0] = 255; m[0:5, :, 0] = 0
    r = check_mask_quality(m, (10, 10))
    assert r.ok
    assert abs(r.excluded_ratio - 0.5) < 1e-6
