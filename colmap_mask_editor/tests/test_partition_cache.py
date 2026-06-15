"""V0.9: partition_builder の end-to-end 生成・キャッシュ有効判定・stale 検出。"""

import cv2
import numpy as np
import pytest

from ai import partition_npz
from ai import partition_manifest as pman
from ai.amg_manifest import read_json, source_fingerprint
from partition_backend import partition_builder as builder

from tests._partition_helpers import synthetic_bgr


def _write_image(tmp_path):
    img = synthetic_bgr(120, 160, seed=4)
    p = tmp_path / "サブ フォルダ" / "IMG_0001.png"   # 日本語 + 全角スペース
    p.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", img)
    p.write_bytes(buf.tobytes())
    return p


SETTINGS = {
    "backend": "auto", "working_max_side": 0, "base_region_count": 50,
    "default_visible_count": 20, "min_region_area_ratio": 10,
    "weight_color": 30, "weight_texture": 10, "weight_boundary": 30,
    "weight_sam": 25, "weight_size": 5, "sam_sample_count": 32, "sam_top_k": 4,
}


def test_build_partition_end_to_end(tmp_path):
    img_path = _write_image(tmp_path)
    out = tmp_path / "cache"
    stages = []
    manifest = builder.build_partition(
        img_path, image_key="サブ/IMG_0001.png", output_dir=out,
        settings=SETTINGS, progress=lambda s, f, i: stages.append(s))
    assert manifest["coverage"]["coverage_ratio"] == 1.0
    assert manifest["coverage"]["unassigned_pixels"] == 0
    assert manifest["backend_used"] == "grid_watershed"  # ximgproc 無し環境
    assert "completed" in stages
    # 生成物が揃う
    assert (out / pman.PARTITION_NPZ_NAME).exists()
    assert (out / pman.PARTITION_MANIFEST_NAME).exists()
    assert (out / pman.PARTITION_REVIEW_NAME).exists()
    # npz 再検証
    partition_npz.verify_partition_npz(out / pman.PARTITION_NPZ_NAME)


def test_cache_valid_when_unchanged(tmp_path):
    img_path = _write_image(tmp_path)
    out = tmp_path / "cache"
    builder.build_partition(img_path, image_key="k", output_dir=out, settings=SETTINGS)
    manifest = read_json(out / pman.PARTITION_MANIFEST_NAME)
    sha = partition_npz.file_sha256(out / pman.PARTITION_NPZ_NAME)
    valid, reason = pman.partition_cache_status(
        manifest, source_fingerprint=source_fingerprint(img_path),
        original_width=160, original_height=120,
        segments_npz_sha256=None, partition_npz_sha256=sha,
        settings_hash=pman.partition_settings_hash(SETTINGS))
    assert valid, reason


def test_cache_stale_on_settings_change(tmp_path):
    img_path = _write_image(tmp_path)
    out = tmp_path / "cache"
    builder.build_partition(img_path, image_key="k", output_dir=out, settings=SETTINGS)
    manifest = read_json(out / pman.PARTITION_MANIFEST_NAME)
    sha = partition_npz.file_sha256(out / pman.PARTITION_NPZ_NAME)
    changed = dict(SETTINGS, weight_color=99)
    valid, reason = pman.partition_cache_status(
        manifest, source_fingerprint=source_fingerprint(img_path),
        original_width=160, original_height=120,
        segments_npz_sha256=None, partition_npz_sha256=sha,
        settings_hash=pman.partition_settings_hash(changed))
    assert not valid


def test_review_update_does_not_touch_npz(tmp_path):
    img_path = _write_image(tmp_path)
    out = tmp_path / "cache"
    builder.build_partition(img_path, image_key="k", output_dir=out, settings=SETTINGS)
    npz_path = out / pman.PARTITION_NPZ_NAME
    before = partition_npz.file_sha256(npz_path)
    pman.update_partition_review(
        out / pman.PARTITION_REVIEW_NAME,
        node_decisions={"1": "keep"}, completed=False)
    after = partition_npz.file_sha256(npz_path)
    assert before == after  # partition.npz は不変


def test_cancel_keeps_no_partial(tmp_path):
    img_path = _write_image(tmp_path)
    out = tmp_path / "cache"
    with pytest.raises(builder.PartitionCancelled):
        builder.build_partition(
            img_path, image_key="k", output_dir=out, settings=SETTINGS,
            should_cancel=lambda: True)  # 即キャンセル
    # partition.npz は作られていない (保存前に中断)
    assert not (out / pman.PARTITION_NPZ_NAME).exists()
