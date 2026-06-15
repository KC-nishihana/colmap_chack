"""V0.8: segments.npz の決定的ビルド・原子保存・検証テスト (torch 不要)。"""

import numpy as np
import pytest

from ai import amg_npz, amg_rle
from ai.amg_npz import NpzValidationError


def _ann_from_mask(m, iou=0.9, stab=0.95):
    h, w = m.shape
    counts = amg_rle.encode_mask(m)
    ys, xs = np.where(m > 0)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
    return {
        "segmentation": {"size": [h, w], "counts": counts},
        "area": int((m > 0).sum()),
        "bbox": bbox,
        "predicted_iou": iou,
        "stability_score": stab,
        "point_coords": [[float(xs.mean()), float(ys.mean())]],
        "crop_box": [0, 0, w, h],
    }


def _sample_annotations(h=20, w=24):
    small = np.zeros((h, w), np.uint8); small[1:3, 1:3] = 1
    big = np.zeros((h, w), np.uint8); big[2:18, 2:20] = 1
    mid = np.zeros((h, w), np.uint8); mid[5:12, 5:14] = 1
    return [_ann_from_mask(small), _ann_from_mask(big), _ann_from_mask(mid)], h, w


def test_build_segment_arrays_deterministic_sort():
    anns, h, w = _sample_annotations()
    arrays = amg_npz.build_segment_arrays(anns, h, w)
    # area 降順 -> 最初は big
    areas = arrays["area"].tolist()
    assert areas == sorted(areas, reverse=True)
    assert arrays["segment_ids"].tolist() == [1, 2, 3]


def test_save_load_verify_roundtrip(tmp_path):
    anns, h, w = _sample_annotations()
    arrays = amg_npz.build_segment_arrays(anns, h, w)
    path = tmp_path / "segments.npz"
    sha = amg_npz.save_segments_npz(path, arrays)
    assert len(sha) == 64
    data = amg_npz.verify_segments_npz(path)
    assert data["segment_ids"].shape == (3,)
    # round trip: decode segment 0 area == area[0]
    c0 = amg_rle.unpack_counts(data, 0)
    assert amg_rle.rle_area(c0) == int(data["area"][0])


def test_npz_loads_without_pickle(tmp_path):
    anns, h, w = _sample_annotations()
    arrays = amg_npz.build_segment_arrays(anns, h, w)
    path = tmp_path / "segments.npz"
    amg_npz.save_segments_npz(path, arrays)
    with np.load(path, allow_pickle=False) as data:
        assert "rle_counts" in data.files


def test_npz_has_no_dense_mask(tmp_path):
    anns, h, w = _sample_annotations()
    arrays = amg_npz.build_segment_arrays(anns, h, w)
    path = tmp_path / "segments.npz"
    amg_npz.save_segments_npz(path, arrays)
    with np.load(path, allow_pickle=False) as data:
        for name in data.files:
            assert data[name].ndim < 3, f"{name} が dense マスク"


def test_dtypes_match_schema(tmp_path):
    anns, h, w = _sample_annotations()
    arrays = amg_npz.build_segment_arrays(anns, h, w)
    path = tmp_path / "segments.npz"
    amg_npz.save_segments_npz(path, arrays)
    data = amg_npz.load_segments_npz(path)
    for name, dtype in amg_npz.REQUIRED_ARRAYS.items():
        assert data[name].dtype == dtype, f"{name}: {data[name].dtype} != {dtype}"


def test_atomic_no_leftover_tmp(tmp_path):
    anns, h, w = _sample_annotations()
    arrays = amg_npz.build_segment_arrays(anns, h, w)
    path = tmp_path / "segments.npz"
    amg_npz.save_segments_npz(path, arrays)
    assert not (tmp_path / "segments.npz.tmp").exists()


def test_corrupt_detection(tmp_path):
    anns, h, w = _sample_annotations()
    arrays = amg_npz.build_segment_arrays(anns, h, w)
    path = tmp_path / "segments.npz"
    amg_npz.save_segments_npz(path, arrays)
    # ファイルを破壊
    raw = bytearray(path.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    path.write_bytes(bytes(raw))
    with pytest.raises((NpzValidationError, Exception)):
        amg_npz.verify_segments_npz(path)


def test_verify_rejects_tampered_area(tmp_path):
    anns, h, w = _sample_annotations()
    arrays = amg_npz.build_segment_arrays(anns, h, w)
    arrays["area"] = arrays["area"].copy()
    arrays["area"][0] = int(arrays["area"][0]) + 5  # 改ざん
    path = tmp_path / "segments.npz"
    # save 時の verify で弾かれる
    with pytest.raises(NpzValidationError):
        amg_npz.save_segments_npz(path, arrays)


def test_empty_segments(tmp_path):
    arrays = amg_npz.build_segment_arrays([], 10, 10)
    path = tmp_path / "segments.npz"
    amg_npz.save_segments_npz(path, arrays)
    data = amg_npz.verify_segments_npz(path)
    assert data["segment_ids"].shape == (0,)
