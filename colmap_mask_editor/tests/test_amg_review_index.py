"""V0.10: review_index.npz の構築・原子保存・検証・stale 判定のテスト。"""

import numpy as np
import pytest

from ai import amg_npz, amg_rle, amg_review_index as ri


def _ann(m, iou=0.9, stab=0.95):
    h, w = m.shape
    ys, xs = np.where(m > 0)
    return {
        "segmentation": {"size": [h, w], "counts": amg_rle.encode_mask(m)},
        "area": int((m > 0).sum()),
        "bbox": [int(xs.min()), int(ys.min()),
                 int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
        "predicted_iou": float(iou), "stability_score": float(stab),
        "point_coords": [[float(xs.mean()), float(ys.mean())]], "crop_box": [0, 0, w, h],
    }


def _rect(h, w, y0, y1, x0, x1, **kw):
    m = np.zeros((h, w), np.uint8); m[y0:y1, x0:x1] = 1
    return _ann(m, **kw)


def _arrays():
    h, w = 60, 60
    anns = [
        _rect(h, w, 0, 50, 0, 50, iou=0.9, stab=0.9),    # 大・端接触
        _rect(h, w, 5, 15, 5, 15, iou=0.8, stab=0.8),    # 小・outer 内
        _rect(h, w, 40, 50, 40, 50, iou=0.7, stab=0.95), # 別対象
    ]
    return amg_npz.build_segment_arrays(anns, h, w)


def test_build_arrays_schema():
    arrays = ri.build_review_index_arrays(_arrays())
    for name, dtype in ri.REQUIRED_ARRAYS.items():
        assert name in arrays
        assert arrays[name].dtype == dtype
    n = arrays["segment_ids"].shape[0]
    assert arrays["group_ids"].shape == (n,)
    assert arrays["priority_scores"].shape == (n,)


def test_priority_weights():
    a = _arrays()
    arrays = ri.build_review_index_arrays(a)
    quality = arrays["quality_scores"]
    edge = arrays["edge_touch_flags"]
    area = np.asarray(a["area"], dtype=np.float64)
    max_area = area.max()
    expected = (ri.W_AREA * (area / max_area)
                + ri.W_QUALITY * quality + ri.W_EDGE * edge)
    assert np.allclose(arrays["priority_scores"], expected.astype(np.float32), atol=1e-6)


def test_save_load_roundtrip_no_pickle(tmp_path):
    arrays = ri.build_review_index_arrays(_arrays())
    path = tmp_path / "review_index.npz"
    ri.save_review_index(path, arrays)
    # allow_pickle=False で読めること
    loaded = ri.load_review_index(path)
    assert np.array_equal(loaded["segment_ids"], arrays["segment_ids"])
    with np.load(path, allow_pickle=False) as d:
        assert "group_ids" in d.files


def test_verify_rejects_dense(tmp_path):
    arrays = ri.build_review_index_arrays(_arrays())
    arrays["bad_dense"] = np.zeros((4, 4), dtype=np.uint8)
    path = tmp_path / "ri.npz"
    with pytest.raises(ri.ReviewIndexError):
        ri.save_review_index(path, arrays)


def test_stale_on_sha_change():
    m = ri.build_review_index_manifest(
        segments_npz_sha256="abc", settings_hash="h1",
        group_count=2, segment_count=3)
    assert not ri.is_review_index_stale(m, segments_npz_sha256="abc", settings_hash="h1")
    assert ri.is_review_index_stale(m, segments_npz_sha256="DIFF", settings_hash="h1")


def test_stale_on_threshold_change():
    h1 = ri.grouping_settings_hash(0.85, 0.95)
    h2 = ri.grouping_settings_hash(0.80, 0.95)
    assert h1 != h2
    m = ri.build_review_index_manifest(
        segments_npz_sha256="abc", settings_hash=h1, group_count=2, segment_count=3)
    assert ri.is_review_index_stale(m, segments_npz_sha256="abc", settings_hash=h2)


def test_stale_when_missing_manifest():
    assert ri.is_review_index_stale(None, segments_npz_sha256="x", settings_hash="y")
    assert ri.is_review_index_stale({}, segments_npz_sha256="x", settings_hash="y")


def test_settings_hash_deterministic():
    assert ri.grouping_settings_hash(0.85, 0.95) == ri.grouping_settings_hash(0.85, 0.95)


# ---------- ensure_review_index (Worker の純粋関数) ----------

def _build_cache(tmp_path):
    arrays = _arrays()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    from ai import amg_manifest
    amg_npz.save_segments_npz(cache_dir / amg_manifest.SEGMENTS_NPZ_NAME, arrays)
    return cache_dir


def test_ensure_builds_then_reuses(tmp_path):
    from core.amg_review_index_worker import ensure_review_index
    cache_dir = _build_cache(tmp_path)
    r1 = ensure_review_index(cache_dir)
    assert r1.status == "built"
    assert (cache_dir / ri.REVIEW_INDEX_NPZ_NAME).exists()
    assert (cache_dir / ri.REVIEW_INDEX_MANIFEST_NAME).exists()
    r2 = ensure_review_index(cache_dir)
    assert r2.status == "reused"
    assert r2.group_count == r1.group_count


def test_ensure_force_rebuilds(tmp_path):
    from core.amg_review_index_worker import ensure_review_index
    cache_dir = _build_cache(tmp_path)
    ensure_review_index(cache_dir)
    r = ensure_review_index(cache_dir, force=True)
    assert r.status == "built"


def test_ensure_rebuilds_on_threshold_change(tmp_path):
    from core.amg_review_index_worker import ensure_review_index
    cache_dir = _build_cache(tmp_path)
    ensure_review_index(cache_dir, iou_threshold=0.85)
    r = ensure_review_index(cache_dir, iou_threshold=0.50)
    assert r.status == "built"


def test_ensure_cancellable(tmp_path):
    from core.amg_review_index_worker import ensure_review_index, ReviewIndexCancelled
    cache_dir = _build_cache(tmp_path)
    with pytest.raises(ReviewIndexCancelled):
        ensure_review_index(cache_dir, cancel_check=lambda: True)
