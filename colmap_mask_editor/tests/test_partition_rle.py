"""V0.9: 葉 region map の C-order RLE encode/decode/point-lookup テスト。"""

import numpy as np
import pytest

from ai import partition_rle
from ai.partition_rle import RegionRleError


def test_roundtrip_simple():
    labels = np.array([[1, 1, 2], [2, 2, 3]], dtype=np.int32)
    ids, lengths = partition_rle.encode_label_map(labels)
    out = partition_rle.decode_to_label_map(ids, lengths, 2, 3)
    assert np.array_equal(out, labels)


def test_runs_are_merged():
    labels = np.array([[1, 1, 1, 1]], dtype=np.int32)
    ids, lengths = partition_rle.encode_label_map(labels)
    assert ids.tolist() == [1]
    assert lengths.tolist() == [4]


def test_run_length_sum_equals_pixels():
    rng = np.random.default_rng(0)
    labels = rng.integers(1, 6, size=(20, 30)).astype(np.int32)
    ids, lengths = partition_rle.encode_label_map(labels)
    assert int(lengths.sum()) == 20 * 30
    assert np.all(ids >= 1)


def test_dtypes():
    labels = np.array([[1, 2], [3, 4]], dtype=np.int32)
    ids, lengths = partition_rle.encode_label_map(labels)
    assert ids.dtype == np.uint32
    assert lengths.dtype == np.uint64


def test_point_lookup_matches_decode():
    rng = np.random.default_rng(1)
    labels = rng.integers(1, 8, size=(15, 23)).astype(np.int32)
    ids, lengths = partition_rle.encode_label_map(labels)
    h, w = labels.shape
    cum = np.cumsum(lengths.astype(np.int64))
    for y in range(h):
        for x in range(w):
            got = partition_rle.region_at_point(ids, lengths, w, x, y, cum=cum)
            assert got == int(labels[y, x]), (x, y)


def test_point_lookup_c_order_flat_index():
    labels = np.arange(1, 13).reshape(3, 4).astype(np.int32)
    ids, lengths = partition_rle.encode_label_map(labels)
    # flat_index = y*w + x
    assert partition_rle.region_at_index(ids, lengths, 0) == 1
    assert partition_rle.region_at_index(ids, lengths, 11) == 12
    assert partition_rle.region_at_point(ids, lengths, 4, 1, 0) == 2


def test_validate_rejects_zero_region_id():
    ids = np.array([0, 1], dtype=np.uint32)
    lengths = np.array([2, 2], dtype=np.uint64)
    with pytest.raises(RegionRleError):
        partition_rle.validate_region_rle(ids, lengths, 1, 4, leaf_count=1)


def test_validate_rejects_id_over_leaf_count():
    ids = np.array([1, 5], dtype=np.uint32)
    lengths = np.array([2, 2], dtype=np.uint64)
    with pytest.raises(RegionRleError):
        partition_rle.validate_region_rle(ids, lengths, 1, 4, leaf_count=2)


def test_validate_rejects_unmerged_runs():
    ids = np.array([1, 1], dtype=np.uint32)
    lengths = np.array([2, 2], dtype=np.uint64)
    with pytest.raises(RegionRleError):
        partition_rle.validate_region_rle(ids, lengths, 1, 4, leaf_count=1)


def test_validate_rejects_wrong_total():
    ids = np.array([1, 2], dtype=np.uint32)
    lengths = np.array([2, 2], dtype=np.uint64)
    with pytest.raises(RegionRleError):
        partition_rle.validate_region_rle(ids, lengths, 2, 3, leaf_count=2)


def test_out_of_range_index_raises():
    labels = np.array([[1, 2]], dtype=np.int32)
    ids, lengths = partition_rle.encode_label_map(labels)
    with pytest.raises(IndexError):
        partition_rle.region_at_index(ids, lengths, 99)
