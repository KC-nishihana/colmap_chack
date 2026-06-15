"""V0.8: manifest / batch_manifest の構築・原子更新・プリセット・hash テスト。"""

import json

import pytest

from ai import amg_manifest as M


MODEL = {
    "model_id": "sam2.1_hiera_small",
    "sam2_commit": "abc123",
    "checkpoint_fingerprint": "deadbeef",
}


def test_cache_id_stable_and_len():
    cid = M.cache_id_for("sub/IMG_0001.jpg")
    assert len(cid) == 16
    assert cid == M.cache_id_for("sub/IMG_0001.jpg")
    assert cid != M.cache_id_for("sub/IMG_0002.jpg")


def test_match_preset():
    assert M.match_preset(M.preset_settings("fast")) == "fast"
    assert M.match_preset(M.preset_settings("standard")) == "standard"
    custom = M.preset_settings("fast")
    custom["points_per_side"] = 24
    assert M.match_preset(custom) == "custom"


def test_settings_hash_changes_with_settings():
    g = M.preset_settings("fast")
    h1 = M.settings_hash(g, MODEL)
    g2 = dict(g); g2["pred_iou_thresh"] = 0.5
    assert M.settings_hash(g2, MODEL) != h1
    # モデル変更でも変わる
    m2 = dict(MODEL); m2["checkpoint_fingerprint"] = "other"
    assert M.settings_hash(g, m2) != h1
    # 順序非依存 (キー順正規化)
    g3 = {k: g[k] for k in reversed(list(g.keys()))}
    assert M.settings_hash(g3, MODEL) == h1


def test_build_image_manifest_decisions_unreviewed():
    g = M.preset_settings("fast")
    man = M.build_image_manifest(
        image_key="sub/IMG_0001.jpg", source_path="C:/p/IMG_0001.jpg",
        width=100, height=80, model=MODEL, generator=g, preset="fast",
        segment_count=3, segment_ids=[1, 2, 3], segments_npz_sha256="x" * 64,
        processing_time_sec=1.2, fingerprint={"file_size": 10, "mtime_ns": 20},
    )
    assert man["review"]["decisions"] == {"1": "unreviewed", "2": "unreviewed", "3": "unreviewed"}
    assert man["review"]["completed"] is False
    assert man["generator"]["preset"] == "fast"
    assert man["settings_hash"] == M.settings_hash(g, MODEL)


def test_update_manifest_decisions_atomic(tmp_path):
    g = M.preset_settings("fast")
    man = M.build_image_manifest(
        image_key="k", source_path="C:/p/x.jpg", width=10, height=10,
        model=MODEL, generator=g, preset="fast", segment_count=2,
        segment_ids=[1, 2], segments_npz_sha256="x" * 64, processing_time_sec=1.0,
        fingerprint={"file_size": 1, "mtime_ns": 2},
    )
    path = tmp_path / "manifest.json"
    M.atomic_write_json(path, man)
    M.update_manifest_decisions(path, {"1": "keep", "2": "remove"}, completed=True)
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded["review"]["decisions"] == {"1": "keep", "2": "remove"}
    assert reloaded["review"]["completed"] is True
    assert reloaded["review"]["updated_at"] is not None
    assert not (tmp_path / "manifest.json.tmp").exists()


def test_update_manifest_rejects_unknown_decision(tmp_path):
    g = M.preset_settings("fast")
    man = M.build_image_manifest(
        image_key="k", source_path="C:/p/x.jpg", width=10, height=10,
        model=MODEL, generator=g, preset="fast", segment_count=1,
        segment_ids=[1], segments_npz_sha256="x" * 64, processing_time_sec=1.0,
        fingerprint={"file_size": 1, "mtime_ns": 2},
    )
    path = tmp_path / "manifest.json"
    M.atomic_write_json(path, man)
    with pytest.raises(ValueError):
        M.update_manifest_decisions(path, {"1": "bogus"})


def test_batch_manifest_update(tmp_path):
    path = tmp_path / "batch_manifest.json"
    M.update_batch_image_entry(
        path, "sub/IMG_0001.jpg", cache_id="1a2b", status="ready",
        segment_count=84, last_job_id="amg-xyz",
    )
    M.update_batch_image_entry(
        path, "sub/IMG_0002.jpg", cache_id="3c4d", status="failed", error="boom",
    )
    batch = json.loads(path.read_text(encoding="utf-8"))
    assert batch["images"]["sub/IMG_0001.jpg"]["status"] == "ready"
    assert batch["images"]["sub/IMG_0002.jpg"]["error"] == "boom"
    assert batch["last_job_id"] == "amg-xyz"


def test_batch_manifest_rejects_bad_status(tmp_path):
    path = tmp_path / "batch_manifest.json"
    with pytest.raises(ValueError):
        M.update_batch_image_entry(path, "k", cache_id="x", status="weird")
