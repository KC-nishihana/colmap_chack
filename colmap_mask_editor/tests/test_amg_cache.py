"""V0.8: キャッシュ有効性 / stale / corrupt / processing 回復のテスト。"""

import numpy as np
import pytest

from ai import amg_cache, amg_manifest as M, amg_npz, amg_rle


MODEL = {
    "model_id": "sam2.1_hiera_small",
    "sam2_commit": "abc123",
    "checkpoint_fingerprint": "deadbeef",
}


def _ann(m):
    h, w = m.shape
    ys, xs = np.where(m > 0)
    return {
        "segmentation": {"size": [h, w], "counts": amg_rle.encode_mask(m)},
        "area": int((m > 0).sum()),
        "bbox": [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
        "predicted_iou": 0.9, "stability_score": 0.95,
        "point_coords": [[float(xs.mean()), float(ys.mean())]], "crop_box": [0, 0, w, h],
    }


def _setup_cache(tmp_path, generator):
    """src 画像と cache (npz+manifest) を作って cache_dir, src_path を返す。"""
    src = tmp_path / "日本語 画像" / "IMG 0001.jpg"  # 日本語+全角スペース
    src.parent.mkdir(parents=True)
    src.write_bytes(b"\xff\xd8\xff" + b"0" * 500)  # ダミー
    h, w = 16, 20
    m = np.zeros((h, w), np.uint8); m[2:10, 2:12] = 1
    arrays = amg_npz.build_segment_arrays([_ann(m)], h, w)
    cache_dir = tmp_path / "cache" / M.cache_id_for("k")
    cache_dir.mkdir(parents=True)
    sha = amg_npz.save_segments_npz(cache_dir / "segments.npz", arrays)
    man = M.build_image_manifest(
        image_key="k", source_path=str(src), width=w, height=h, model=MODEL,
        generator=generator, preset="fast", segment_count=1, segment_ids=[1],
        segments_npz_sha256=sha, processing_time_sec=1.0,
    )
    M.atomic_write_json(cache_dir / "manifest.json", man)
    return cache_dir, src, (w, h)


def test_reusable(tmp_path):
    g = M.preset_settings("fast")
    cache_dir, src, (w, h) = _setup_cache(tmp_path, g)
    res = amg_cache.evaluate_cache(cache_dir, source_path=str(src), width=w, height=h,
                                   model=MODEL, generator=g)
    assert res.state == amg_cache.REUSABLE, res.reason


def test_stale_on_source_change(tmp_path):
    g = M.preset_settings("fast")
    cache_dir, src, (w, h) = _setup_cache(tmp_path, g)
    src.write_bytes(b"\xff\xd8\xff" + b"1" * 999)  # 書き換え -> size/mtime 変化
    res = amg_cache.evaluate_cache(cache_dir, source_path=str(src), width=w, height=h,
                                   model=MODEL, generator=g)
    assert res.state == amg_cache.STALE


def test_stale_on_settings_change(tmp_path):
    g = M.preset_settings("fast")
    cache_dir, src, (w, h) = _setup_cache(tmp_path, g)
    g2 = dict(g); g2["pred_iou_thresh"] = 0.5
    res = amg_cache.evaluate_cache(cache_dir, source_path=str(src), width=w, height=h,
                                   model=MODEL, generator=g2)
    assert res.state == amg_cache.STALE


def test_stale_on_model_change(tmp_path):
    g = M.preset_settings("fast")
    cache_dir, src, (w, h) = _setup_cache(tmp_path, g)
    m2 = dict(MODEL); m2["model_id"] = "sam2.1_hiera_base_plus"
    res = amg_cache.evaluate_cache(cache_dir, source_path=str(src), width=w, height=h,
                                   model=m2, generator=g)
    assert res.state == amg_cache.STALE


def test_corrupt_on_sha_mismatch(tmp_path):
    g = M.preset_settings("fast")
    cache_dir, src, (w, h) = _setup_cache(tmp_path, g)
    # NPZ を別内容で上書き (manifest の sha と不一致)
    m = np.ones((h, w), np.uint8)
    arrays = amg_npz.build_segment_arrays([_ann(m)], h, w)
    amg_npz.save_segments_npz(cache_dir / "segments.npz", arrays)
    res = amg_cache.evaluate_cache(cache_dir, source_path=str(src), width=w, height=h,
                                   model=MODEL, generator=g)
    assert res.state == amg_cache.CORRUPT


def test_missing(tmp_path):
    g = M.preset_settings("fast")
    res = amg_cache.evaluate_cache(tmp_path / "nope", source_path="x", width=1, height=1,
                                   model=MODEL, generator=g)
    assert res.state == amg_cache.MISSING


def test_recover_processing_states():
    batch = {
        "active_job_id": "amg-1",
        "images": {
            "a": {"status": "processing", "error": "x", "cache_id": "1"},
            "b": {"status": "ready", "cache_id": "2"},
        },
    }
    out = amg_cache.recover_processing_states(batch)
    assert out["images"]["a"]["status"] == "unprocessed"
    assert out["images"]["b"]["status"] == "ready"
    assert out["active_job_id"] is None
