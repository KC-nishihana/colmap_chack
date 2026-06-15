"""V0.8: AMG 最終マスク一括適用 / ロールバック / 取り消しのテスト (torch 不要)。"""

import numpy as np
import pytest

from ai import amg_manifest as M, amg_mask_composer as mc, amg_npz, amg_rle
from core import amg_apply_worker as A
from core.amg_apply_worker import AmgApplyError, AmgApplyTarget
from core.mask_io import imread_jp, imwrite_jp


MODEL = {"model_id": "x", "sam2_commit": "y", "checkpoint_fingerprint": "z"}


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


def _make_cache(tmp_path, image_key, decisions):
    h, w = 12, 16
    a = np.zeros((h, w), np.uint8); a[1:9, 1:11] = 1   # large -> id 1
    b = np.zeros((h, w), np.uint8); b[6:11, 9:15] = 1  # small -> id 2
    arrays = amg_npz.build_segment_arrays([_ann(a), _ann(b)], h, w)
    cache_dir = tmp_path / "segmentation_cache" / "images" / M.cache_id_for(image_key)
    cache_dir.mkdir(parents=True)
    sha = amg_npz.save_segments_npz(cache_dir / "segments.npz", arrays)
    man = M.build_image_manifest(
        image_key=image_key, source_path=str(tmp_path / image_key), width=w, height=h,
        model=MODEL, generator=M.preset_settings("fast"), preset="fast",
        segment_count=int(arrays["segment_ids"].shape[0]),
        segment_ids=arrays["segment_ids"].tolist(), segments_npz_sha256=sha,
        processing_time_sec=1.0, fingerprint={"file_size": 1, "mtime_ns": 2},
    )
    sids = arrays["segment_ids"].tolist()
    dec = {str(s): "unreviewed" for s in sids}
    dec.update({str(k): v for k, v in decisions.items()})
    man["review"]["decisions"] = dec
    M.atomic_write_json(cache_dir / "manifest.json", man)
    return cache_dir, (h, w), sids


def test_apply_keep_only(tmp_path):
    # keep the large segment (id 1)
    cache_dir, (h, w), sids = _make_cache(tmp_path, "IMG1.jpg", {1: "keep"})
    save_path = tmp_path / "masks" / "IMG1.png"
    target = AmgApplyTarget("IMG1.jpg", str(cache_dir), str(save_path))
    outcome = A.apply_amg_batch([target], mc.MODE_KEEP_ONLY, tmp_path / "backup")
    assert outcome.applied == ["IMG1.jpg"]
    out = imread_jp(save_path)
    assert out is not None and out.max() == 255 and out.min() == 0


def test_apply_exclude_remove_with_existing(tmp_path):
    cache_dir, (h, w), sids = _make_cache(tmp_path, "IMG2.jpg", {2: "remove"})
    save_path = tmp_path / "masks" / "IMG2.png"
    save_path.parent.mkdir(parents=True)
    existing = np.full((h, w), 255, np.uint8)
    imwrite_jp(save_path, existing)
    target = AmgApplyTarget("IMG2.jpg", str(cache_dir), str(save_path))
    A.apply_amg_batch([target], mc.MODE_EXCLUDE_REMOVE, tmp_path / "backup")
    out = imread_jp(save_path)
    rm = amg_rle.decode_rle(amg_rle.unpack_counts(amg_npz.load_segments_npz(cache_dir / "segments.npz"), 1), h, w) > 0
    out_gray = out if out.ndim == 2 else out[:, :, 0]
    assert np.all(out_gray[rm] == 0)


def test_rollback_on_failure(tmp_path):
    # 2 画像: 2 つ目の cache を壊して produce 失敗 -> 1 つ目もロールバック
    cdir1, (h, w), s1 = _make_cache(tmp_path, "A.jpg", {1: "keep"})
    cdir2, _, s2 = _make_cache(tmp_path, "B.jpg", {1: "keep"})
    # B の NPZ を削除して compose を失敗させる
    (cdir2 / "segments.npz").unlink()
    save1 = tmp_path / "masks" / "A.png"
    save2 = tmp_path / "masks" / "B.png"
    save1.parent.mkdir(parents=True)
    imwrite_jp(save1, np.full((h, w), 200, np.uint8))  # 既存マスク (復元対象)
    t1 = AmgApplyTarget("A.jpg", str(cdir1), str(save1))
    t2 = AmgApplyTarget("B.jpg", str(cdir2), str(save2))
    with pytest.raises(AmgApplyError):
        A.apply_amg_batch([t1, t2], mc.MODE_KEEP_ONLY, tmp_path / "backup")
    # A は元の 200 のまま (コミットされない or 復元される)、B は作られない
    out1 = imread_jp(save1)
    out1g = out1 if out1.ndim == 2 else out1[:, :, 0]
    assert int(out1g.max()) == 200
    assert not save2.exists()


def test_undo_restores_and_deletes(tmp_path):
    cdir1, (h, w), s1 = _make_cache(tmp_path, "A.jpg", {1: "keep"})
    cdir2, _, s2 = _make_cache(tmp_path, "B.jpg", {1: "keep"})
    save1 = tmp_path / "masks" / "A.png"  # 既存あり
    save2 = tmp_path / "masks" / "B.png"  # 既存なし (新規作成)
    save1.parent.mkdir(parents=True)
    imwrite_jp(save1, np.full((h, w), 123, np.uint8))
    t1 = AmgApplyTarget("A.jpg", str(cdir1), str(save1))
    t2 = AmgApplyTarget("B.jpg", str(cdir2), str(save2))
    outcome = A.apply_amg_batch([t1, t2], mc.MODE_KEEP_ONLY, tmp_path / "backup")
    assert save2.exists()
    undone = A.undo_amg_batch(outcome.record)
    assert set(undone) == {"A.jpg", "B.jpg"}
    # A は元の 123 へ復元、B は削除
    out1 = imread_jp(save1)
    out1g = out1 if out1.ndim == 2 else out1[:, :, 0]
    assert int(out1g.max()) == 123
    assert not save2.exists()


def test_bad_mode_rejected(tmp_path):
    cdir, _, _ = _make_cache(tmp_path, "A.jpg", {1: "keep"})
    t = AmgApplyTarget("A.jpg", str(cdir), str(tmp_path / "m.png"))
    with pytest.raises(AmgApplyError):
        A.apply_amg_batch([t], "bogus_mode", tmp_path / "backup")


def test_size_mismatch_existing_mask_aborts(tmp_path):
    """既存マスクが解析画像とサイズ不一致なら、黙って無視せず中止する。

    無視すると「不要領域を除外」で全面 255 から処理が始まり、
    既存マスクの内容が静かに失われてしまう。"""
    cache_dir, (h, w), sids = _make_cache(tmp_path, "IMG3.jpg", {2: "remove"})
    save_path = tmp_path / "masks" / "IMG3.png"
    save_path.parent.mkdir(parents=True)
    # サイズの違う既存マスクを置く
    imwrite_jp(save_path, np.full((h + 5, w + 7), 255, np.uint8))
    target = AmgApplyTarget("IMG3.jpg", str(cache_dir), str(save_path))
    with pytest.raises(AmgApplyError, match="サイズが一致しません"):
        A.compose_target_mask(target, mc.MODE_EXCLUDE_REMOVE)
    # トランザクション経由でも中止し、既存マスクは保たれる
    with pytest.raises(AmgApplyError):
        A.apply_amg_batch([target], mc.MODE_EXCLUDE_REMOVE, tmp_path / "backup")
    out = imread_jp(save_path)
    out_g = out if out.ndim == 2 else out[:, :, 0]
    assert out_g.shape == (h + 5, w + 7) and int(out_g.min()) == 255
