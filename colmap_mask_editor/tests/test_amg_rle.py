"""V0.8: NumPy のみの RLE encode/decode/hit-test/validate のテスト (torch 不要)。"""

import numpy as np
import pytest

from ai import amg_rle
from ai.amg_rle import RleError


def _roundtrip(mask):
    counts = amg_rle.encode_mask(mask)
    h, w = mask.shape
    amg_rle.validate_rle(counts, h, w)
    decoded = amg_rle.decode_rle(counts, h, w)
    return counts, decoded


def test_roundtrip_random():
    rng = np.random.default_rng(0)
    mask = (rng.random((37, 53)) > 0.5).astype(np.uint8)
    _, decoded = _roundtrip(mask)
    assert np.array_equal(decoded > 0, mask > 0)


def test_roundtrip_empty_mask():
    mask = np.zeros((10, 8), dtype=np.uint8)
    counts, decoded = _roundtrip(mask)
    assert counts == [80]  # 全背景
    assert decoded.sum() == 0


def test_roundtrip_full_mask():
    mask = np.ones((10, 8), dtype=np.uint8)
    counts, decoded = _roundtrip(mask)
    assert counts == [0, 80]  # 先頭背景0, 前景80
    assert np.all(decoded == 255)


def test_single_pixel():
    mask = np.zeros((5, 6), dtype=np.uint8)
    mask[2, 3] = 1
    counts, decoded = _roundtrip(mask)
    assert np.array_equal(decoded > 0, mask > 0)
    assert amg_rle.rle_area(counts) == 1


def test_counts_starts_with_background():
    # 先頭ピクセルが前景でも counts[0] は背景長 (=0) から始まる
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0, 0] = 1
    counts = amg_rle.encode_mask(mask)
    assert counts[0] == 0


def test_fortran_order_flat_index():
    h, w = 6, 5
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[4, 2] = 1  # (y=4, x=2)
    counts = amg_rle.encode_mask(mask)
    # Fortran order の位置 = y + x*h = 4 + 2*6 = 16
    assert amg_rle.rle_contains_point(counts, h, w, 2, 4) is True
    assert amg_rle.rle_contains_point(counts, h, w, 4, 2) is False


def test_contains_point_out_of_range():
    mask = np.ones((5, 5), dtype=np.uint8)
    counts = amg_rle.encode_mask(mask)
    assert amg_rle.rle_contains_point(counts, 5, 5, -1, 0) is False
    assert amg_rle.rle_contains_point(counts, 5, 5, 0, 5) is False
    assert amg_rle.rle_contains_point(counts, 5, 5, 5, 0) is False


def test_contains_point_matches_decode():
    rng = np.random.default_rng(2)
    h, w = 23, 19
    mask = (rng.random((h, w)) > 0.6).astype(np.uint8)
    counts = amg_rle.encode_mask(mask)
    decoded = amg_rle.decode_rle(counts, h, w) > 0
    for _ in range(200):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        assert amg_rle.rle_contains_point(counts, h, w, x, y) == bool(decoded[y, x])


def test_validate_rejects_negative():
    with pytest.raises(RleError):
        amg_rle.validate_rle([5, -1, 4], 2, 5)


def test_validate_rejects_sum_mismatch():
    with pytest.raises(RleError):
        amg_rle.validate_rle([5, 3], 4, 4)  # 合計8 != 16


def test_validate_rejects_2d():
    with pytest.raises(RleError):
        amg_rle.validate_rle(np.zeros((2, 2)), 2, 2)


def test_decode_dtype_and_values():
    mask = np.zeros((3, 3), dtype=np.uint8)
    mask[1, 1] = 1
    counts = amg_rle.encode_mask(mask)
    decoded = amg_rle.decode_rle(counts, 3, 3)
    assert decoded.dtype == np.uint8
    assert set(np.unique(decoded).tolist()) <= {0, 255}


def test_area_matches_foreground_runs():
    rng = np.random.default_rng(7)
    mask = (rng.random((40, 40)) > 0.5).astype(np.uint8)
    counts = amg_rle.encode_mask(mask)
    assert amg_rle.rle_area(counts) == int((mask > 0).sum())


def test_8k_scale_performance():
    # 8K 相当の縦縞でも contains_point が全復号せず動く
    h, w = 4320, 7680
    counts = []
    # 交互の列ストライプ: 各列 h 個。背景列, 前景列, ...
    parity_fg = False
    run = 0
    for x in range(w):
        col_fg = (x % 2 == 1)
        if col_fg == parity_fg:
            run += h
        else:
            counts.append(run)
            parity_fg = col_fg
            run = h
    counts.append(run)
    amg_rle.validate_rle(counts, h, w)
    # 前景列 x=1 の任意 y は前景, 背景列 x=0 は背景
    assert amg_rle.rle_contains_point(counts, h, w, 1, 100) is True
    assert amg_rle.rle_contains_point(counts, h, w, 0, 100) is False


def test_pack_rles_basic():
    h, w = 10, 12
    m1 = np.zeros((h, w), np.uint8); m1[0:4, 0:4] = 1
    m2 = np.zeros((h, w), np.uint8); m2[5:9, 6:10] = 1
    anns = []
    for m in (m1, m2):
        counts = amg_rle.encode_mask(m)
        ys, xs = np.where(m > 0)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1)]
        anns.append({
            "segmentation": {"size": [h, w], "counts": counts},
            "area": int((m > 0).sum()),
            "bbox": bbox,
            "predicted_iou": 0.9,
            "stability_score": 0.95,
            "point_coords": [[float(xs.mean()), float(ys.mean())]],
            "crop_box": [0, 0, w, h],
        })
    packed = amg_rle.pack_rles(anns, h, w)
    assert packed["rle_offsets"].shape == (3,)
    assert int(packed["rle_offsets"][-1]) == packed["rle_counts"].shape[0]
    c0 = amg_rle.unpack_counts(packed, 0)
    assert amg_rle.rle_area(c0) == int((m1 > 0).sum())


# --- SAM2 公式 uncompressed RLE との互換性 (torch/sam2 が無ければ skip) ---


def test_sam2_uncompressed_rle_compat():
    torch = pytest.importorskip("torch")
    amg = pytest.importorskip("sam2.utils.amg")
    rng = np.random.default_rng(11)
    h, w = 31, 27
    mask = (rng.random((h, w)) > 0.5)
    t = torch.from_numpy(mask[None, ...])
    sam_rle = amg.mask_to_rle_pytorch(t)[0]
    sam_counts = sam_rle["counts"]
    assert sam_rle["size"] == [h, w]
    # 我々の encode と SAM2 の counts が一致
    our_counts = amg_rle.encode_mask(mask.astype(np.uint8))
    assert our_counts == [int(c) for c in sam_counts]
    # SAM2 の decode と我々の decode が一致
    sam_decoded = amg.rle_to_mask(sam_rle)
    our_decoded = amg_rle.decode_rle(sam_counts, h, w) > 0
    assert np.array_equal(sam_decoded, our_decoded)
    # area も一致
    assert amg_rle.rle_area(sam_counts) == amg.area_from_rle(sam_rle)
