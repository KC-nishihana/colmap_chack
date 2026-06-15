"""V0.10: RLE 同士の重なり計算 (dense 復号なし) のテスト。"""

import numpy as np

from ai import amg_rle, amg_rle_overlap as ov


def _counts(mask: np.ndarray):
    return amg_rle.encode_mask(mask)


def _dense_inter(a, b):
    return int((a.astype(bool) & b.astype(bool)).sum())


def _dense_union(a, b):
    return int((a.astype(bool) | b.astype(bool)).sum())


def test_identical_masks():
    m = np.zeros((10, 12), np.uint8); m[2:8, 3:9] = 1
    ca = cb = _counts(m)
    area = int((m > 0).sum())
    assert ov.rle_intersection_area(ca, cb) == area
    assert ov.rle_union_area(ca, cb) == area
    assert ov.rle_iou(ca, cb) == 1.0
    assert ov.rle_containment(ca, cb) == 1.0


def test_no_overlap():
    a = np.zeros((10, 10), np.uint8); a[0:3, 0:3] = 1
    b = np.zeros((10, 10), np.uint8); b[7:10, 7:10] = 1
    ca, cb = _counts(a), _counts(b)
    assert ov.rle_intersection_area(ca, cb) == 0
    assert ov.rle_union_area(ca, cb) == _dense_union(a, b)
    assert ov.rle_iou(ca, cb) == 0.0
    assert ov.rle_containment(ca, cb) == 0.0


def test_partial_containment():
    outer = np.zeros((10, 10), np.uint8); outer[0:8, 0:8] = 1
    inner = np.zeros((10, 10), np.uint8); inner[2:6, 2:6] = 1   # 完全に outer 内
    co, ci = _counts(outer), _counts(inner)
    assert ov.rle_containment(ci, co) == 1.0       # inner は outer に完全包含
    assert ov.rle_containment(co, ci) < 1.0        # 逆は包含されない
    assert ov.rle_intersection_area(ci, co) == int((inner > 0).sum())


def test_same_area_partial_overlap():
    a = np.zeros((10, 10), np.uint8); a[0:5, 0:6] = 1   # area 30
    b = np.zeros((10, 10), np.uint8); b[0:6, 0:5] = 1   # area 30
    ca, cb = _counts(a), _counts(b)
    assert ov.rle_area(ca) == ov.rle_area(cb) == 30
    assert ov.rle_intersection_area(ca, cb) == _dense_inter(a, b)
    assert ov.rle_union_area(ca, cb) == _dense_union(a, b)


def test_empty_mask():
    empty = np.zeros((10, 10), np.uint8)
    full = np.ones((10, 10), np.uint8)
    ce, cf = _counts(empty), _counts(full)
    assert ov.rle_intersection_area(ce, cf) == 0
    assert ov.rle_union_area(ce, cf) == 100
    assert ov.rle_iou(ce, cf) == 0.0
    assert ov.rle_containment(ce, cf) == 0.0     # area(inner=empty)=0 -> 0.0


def test_full_mask():
    full = np.ones((8, 8), np.uint8)
    cf = _counts(full)
    assert ov.rle_intersection_area(cf, cf) == 64
    assert ov.rle_union_area(cf, cf) == 64
    assert ov.rle_iou(cf, cf) == 1.0


def test_fortran_order_consistency():
    # 列優先で復号される RLE と一致することを確認 (非対称な形状)
    a = np.zeros((6, 14), np.uint8); a[1:5, 2:13] = 1
    b = np.zeros((6, 14), np.uint8); b[3:6, 0:8] = 1
    ca, cb = _counts(a), _counts(b)
    assert ov.rle_intersection_area(ca, cb) == _dense_inter(a, b)
    assert ov.rle_union_area(ca, cb) == _dense_union(a, b)
    da = amg_rle.decode_rle(ca, 6, 14) > 0
    assert np.array_equal(da, a > 0)


def test_matches_dense_random():
    rng = np.random.default_rng(1234)
    h, w = 40, 50
    for _ in range(30):
        a = (rng.random((h, w)) < 0.4).astype(np.uint8)
        b = (rng.random((h, w)) < 0.4).astype(np.uint8)
        ca, cb = _counts(a), _counts(b)
        assert ov.rle_intersection_area(ca, cb) == _dense_inter(a, b)
        assert ov.rle_union_area(ca, cb) == _dense_union(a, b)
        ai = int((a > 0).sum())
        if ai:
            exp = _dense_inter(a, b) / ai
            assert abs(ov.rle_containment(ca, cb) - exp) < 1e-9


def test_8k_scale_run_length():
    # 8K 相当の縦ストライプを持つマスクで run 長が多くても正しく動く
    h, w = 4320, 1000   # 高さ大 -> Fortran order で run が多数
    a = np.zeros((h, w), np.uint8); a[:, 0:400] = 1
    b = np.zeros((h, w), np.uint8); b[:, 300:700] = 1
    ca, cb = _counts(a), _counts(b)
    assert ov.rle_intersection_area(ca, cb) == h * 100   # 列 300..399
    assert ov.rle_union_area(ca, cb) == h * 700
