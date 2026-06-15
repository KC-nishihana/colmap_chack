"""V0.8: 日本語・全角スペースを含む Windows パスでの NPZ/manifest 保存テスト。"""

import json

import numpy as np

from ai import amg_manifest as M, amg_npz, amg_rle


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


def test_npz_save_under_unicode_path(tmp_path):
    base = tmp_path / "プロジェクト 名" / "segmentation_cache" / "images" / "1a2b3c4d5e6f7890"
    base.mkdir(parents=True)
    m = np.zeros((12, 14), np.uint8); m[1:6, 1:6] = 1
    arrays = amg_npz.build_segment_arrays([_ann(m)], 12, 14)
    sha = amg_npz.save_segments_npz(base / "segments.npz", arrays)
    assert len(sha) == 64
    data = amg_npz.verify_segments_npz(base / "segments.npz")
    assert data["segment_ids"].shape == (1,)


def test_manifest_unicode_image_key(tmp_path):
    base = tmp_path / "全角　スペース"  # 全角スペース入り
    base.mkdir(parents=True)
    g = M.preset_settings("fast")
    man = M.build_image_manifest(
        image_key="サブ/画像 001.jpg", source_path=str(base / "画像 001.jpg"),
        width=10, height=10, model={"model_id": "x", "sam2_commit": "y", "checkpoint_fingerprint": "z"},
        generator=g, preset="fast", segment_count=1, segment_ids=[1],
        segments_npz_sha256="a" * 64, processing_time_sec=1.0,
        fingerprint={"file_size": 1, "mtime_ns": 2},
    )
    path = base / "manifest.json"
    M.atomic_write_json(path, man)
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded["image_key"] == "サブ/画像 001.jpg"


def test_cache_id_unicode_key_ascii_dir():
    cid = M.cache_id_for("サブ/画像 001.jpg")
    assert len(cid) == 16
    assert cid.isalnum()  # フォルダ名は ASCII 16 進
