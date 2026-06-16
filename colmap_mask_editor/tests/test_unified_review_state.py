"""V0.11: 統合レビュー状態 (unified_review.json) のテスト。

通常マスクが正本・状態 JSON は補助。マスク SHA 不一致で stale。原子保存。
"""

import json

import numpy as np
import pytest

from ai import unified_review_state as urs


def _mask(h=10, w=12, fill=255):
    return np.full((h, w), fill, np.uint8)


def test_mask_sha_changes_with_content():
    a = _mask(); b = _mask()
    assert urs.compute_mask_sha256(a) == urs.compute_mask_sha256(b)
    b[0, 0] = 0
    assert urs.compute_mask_sha256(a) != urs.compute_mask_sha256(b)


def test_build_state_normalizes_actions():
    st = urs.build_unified_review_state(
        image_key="k", segments_npz_sha256="npz", mask_sha256="m",
        candidate_actions={"12": "remove", 35: "add"})
    assert st["schema_version"] == urs.SCHEMA_VERSION
    assert st["candidate_actions"] == {"12": "remove", "35": "add"}
    assert st["completed"] is False


def test_build_state_rejects_unknown_action():
    with pytest.raises(ValueError):
        urs.build_unified_review_state(
            image_key="k", segments_npz_sha256="n", mask_sha256="m",
            candidate_actions={"1": "keep"})


def test_actions_filtered_by_valid_ids():
    actions = {"12": "remove", "999": "add"}
    out = urs.normalize_candidate_actions(actions, valid_segment_ids=[12, 35])
    assert out == {"12": "remove"}          # 999 は不変ID外なので捨てる


def test_save_load_roundtrip_atomic(tmp_path):
    path = tmp_path / "unified_review.json"
    st = urs.build_unified_review_state(
        image_key="サブ/画像 001.png", segments_npz_sha256="npz", mask_sha256="m",
        candidate_actions={"12": "remove"}, ui={"sort_mode": "area"})
    urs.save_unified_review_state(path, st)
    assert not (tmp_path / "unified_review.json.tmp").exists()
    loaded = urs.load_unified_review_state(path)
    assert loaded["image_key"] == "サブ/画像 001.png"
    assert loaded["candidate_actions"] == {"12": "remove"}
    assert loaded["ui"]["sort_mode"] == "area"


def test_load_missing_returns_none(tmp_path):
    assert urs.load_unified_review_state(tmp_path / "nope.json") is None


def test_load_corrupt_returns_none(tmp_path):
    p = tmp_path / "unified_review.json"
    p.write_text("{ broken", encoding="utf-8")
    assert urs.load_unified_review_state(p) is None


def test_stale_on_mask_sha_change():
    st = urs.build_unified_review_state(
        image_key="k", segments_npz_sha256="npz", mask_sha256="m1")
    assert not urs.is_state_stale(st, mask_sha256="m1")
    assert urs.is_state_stale(st, mask_sha256="DIFFERENT")   # マスクが変わった -> stale


def test_stale_on_npz_change_and_missing():
    st = urs.build_unified_review_state(
        image_key="k", segments_npz_sha256="npz", mask_sha256="m1")
    assert urs.is_state_stale(st, mask_sha256="m1", segments_npz_sha256="OTHER")
    assert urs.is_state_stale(None, mask_sha256="m1")


def test_set_and_clear_candidate_action():
    st = urs.build_unified_review_state(
        image_key="k", segments_npz_sha256="n", mask_sha256="m",
        candidate_actions={"12": "remove"})
    st2 = urs.set_candidate_action(st, 35, "add")
    assert st2["candidate_actions"] == {"12": "remove", "35": "add"}
    assert st["candidate_actions"] == {"12": "remove"}   # 元は不変
    st3 = urs.set_candidate_action(st2, 12, None)         # 解除
    assert st3["candidate_actions"] == {"35": "add"}


def test_set_unknown_action_raises():
    st = urs.build_unified_review_state(
        image_key="k", segments_npz_sha256="n", mask_sha256="m")
    with pytest.raises(ValueError):
        urs.set_candidate_action(st, 1, "replace")
