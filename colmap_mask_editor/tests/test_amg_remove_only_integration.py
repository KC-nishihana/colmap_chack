"""
V0.10: REMOVE_ONLY のエンドツーエンド統合テスト。

実 segments.npz を保存し、manifest に remove_only レビューを書き、既存の
core.amg_apply_worker.compose_target_mask (= compose_final_mask の MODE_EXCLUDE_REMOVE)
で最終マスクを生成できることを確認する。基準マスク (全面 / 既存) の両方を検証する。
"""

import cv2
import numpy as np
import pytest

from ai import amg_manifest as M, amg_npz, amg_rle
from core.amg_apply_worker import AmgApplyError, AmgApplyTarget, compose_target_mask


def _ann(m):
    h, w = m.shape
    ys, xs = np.where(m > 0)
    return {
        "segmentation": {"size": [h, w], "counts": amg_rle.encode_mask(m)},
        "area": int((m > 0).sum()),
        "bbox": [int(xs.min()), int(ys.min()),
                 int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
        "predicted_iou": 0.9, "stability_score": 0.95,
        "point_coords": [[float(xs.mean()), float(ys.mean())]], "crop_box": [0, 0, w, h],
    }


def _build(tmp_path, with_existing=False):
    h, w = 30, 40
    a = np.zeros((h, w), np.uint8); a[2:12, 2:18] = 1
    b = np.zeros((h, w), np.uint8); b[15:28, 20:38] = 1
    arrays = amg_npz.build_segment_arrays([_ann(a), _ann(b)], h, w)
    key = "サブ フォルダ/画像 001.png"   # 日本語 + 全角スペース
    cache_dir = tmp_path / "segmentation_cache" / "images" / M.cache_id_for(key)
    cache_dir.mkdir(parents=True)
    src = tmp_path / "images" / "画像 001.png"
    src.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((h, w, 3), 100, np.uint8)
    ok, buf = cv2.imencode(".png", img); buf.tofile(str(src))
    sha = amg_npz.save_segments_npz(cache_dir / "segments.npz", arrays)
    man = M.build_image_manifest(
        image_key=key, source_path=str(src), width=w, height=h,
        model={"model_id": "m", "sam2_commit": "c", "checkpoint_fingerprint": "f"},
        generator=M.preset_settings("fast"), preset="fast",
        segment_count=2, segment_ids=arrays["segment_ids"].tolist(),
        segments_npz_sha256=sha, processing_time_sec=1.0,
        fingerprint={"file_size": 1, "mtime_ns": 2})
    M.atomic_write_json(cache_dir / "manifest.json", man)

    save_path = tmp_path / "masks" / "画像 001.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if with_existing:
        existing = np.zeros((h, w), np.uint8); existing[0:20, :] = 255
        ok, buf = cv2.imencode(".png", existing); buf.tofile(str(save_path))
    return key, cache_dir, save_path, arrays, (h, w)


def test_full_base_remove_only(tmp_path):
    key, cache_dir, save_path, arrays, (h, w) = _build(tmp_path)
    # segment a (id?) を remove に。bbox 左上 (2,2) の id を探す
    rem_sid = next(int(s) for i, s in enumerate(arrays["segment_ids"].tolist())
                   if list(arrays["bbox_xywh"][i][:2]) == [2, 2])
    M.update_manifest_review(
        cache_dir / "manifest.json",
        decisions={str(rem_sid): "remove"},
        workflow=M.REVIEW_WORKFLOW_REMOVE_ONLY, base_mode="full", completed=True)

    target = AmgApplyTarget(key, str(cache_dir), str(save_path))
    out = compose_target_mask(target, "exclude_remove")   # 既存 MODE_EXCLUDE_REMOVE 再利用
    assert out.shape == (h, w)
    assert out.dtype == np.uint8
    idx = arrays["segment_ids"].tolist().index(rem_sid)
    rm = amg_rle.decode_rle(amg_rle.unpack_counts(arrays, idx), h, w) > 0
    assert np.all(out[rm] == 0)        # remove 部分は 0
    assert np.all(out[~rm] == 255)     # それ以外は全面 255 (基準=全面)


def test_existing_base_remove_only(tmp_path):
    key, cache_dir, save_path, arrays, (h, w) = _build(tmp_path, with_existing=True)
    rem_sid = next(int(s) for i, s in enumerate(arrays["segment_ids"].tolist())
                   if list(arrays["bbox_xywh"][i][:2]) == [2, 2])
    M.update_manifest_review(
        cache_dir / "manifest.json",
        decisions={str(rem_sid): "remove"},
        workflow=M.REVIEW_WORKFLOW_REMOVE_ONLY, base_mode="existing_or_full", completed=True)

    target = AmgApplyTarget(key, str(cache_dir), str(save_path))
    out = compose_target_mask(target, "exclude_remove")
    # 既存マスク (上半分 255) を保持しつつ remove 部分は 0
    idx = arrays["segment_ids"].tolist().index(rem_sid)
    rm = amg_rle.decode_rle(amg_rle.unpack_counts(arrays, idx), h, w) > 0
    assert np.all(out[rm] == 0)
    # 既存が 255 かつ remove でない領域は 255 のまま
    keep_region = (np.zeros((h, w), bool))
    keep_region[0:20, :] = True
    assert np.all(out[keep_region & ~rm] == 255)
    # 既存が 0 の領域 (下) は 0 のまま
    assert np.all(out[20:, :] == 0)


def test_existing_size_mismatch_rejected(tmp_path):
    key, cache_dir, save_path, arrays, (h, w) = _build(tmp_path)
    # サイズ不一致の既存マスクを置く -> compose は中止 (全面 255 へ黙って置換しない)
    bad = np.zeros((h + 5, w), np.uint8)
    ok, buf = cv2.imencode(".png", bad); buf.tofile(str(save_path))
    M.update_manifest_review(
        cache_dir / "manifest.json", decisions={},
        workflow=M.REVIEW_WORKFLOW_REMOVE_ONLY, base_mode="existing_or_full")
    target = AmgApplyTarget(key, str(cache_dir), str(save_path))
    with pytest.raises(AmgApplyError):
        compose_target_mask(target, "exclude_remove")


def test_no_remove_yields_full(tmp_path):
    key, cache_dir, save_path, arrays, (h, w) = _build(tmp_path)
    M.update_manifest_review(
        cache_dir / "manifest.json", decisions={},
        workflow=M.REVIEW_WORKFLOW_REMOVE_ONLY, base_mode="full", completed=True)
    target = AmgApplyTarget(key, str(cache_dir), str(save_path))
    out = compose_target_mask(target, "exclude_remove")
    assert np.all(out == 255)   # 候補未選択 -> 全面 255


def test_legacy_manifest_without_workflow_reads_as_standard(tmp_path):
    key, cache_dir, save_path, arrays, (h, w) = _build(tmp_path)
    man = M.read_json(cache_dir / "manifest.json")
    assert "workflow" not in man["review"]           # 従来データ
    assert M.get_review_workflow(man) == M.REVIEW_WORKFLOW_STANDARD
