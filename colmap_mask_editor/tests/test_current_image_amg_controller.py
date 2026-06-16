"""V0.11: 現在画像 AMG コントローラ (動線2 のデータ基盤) のテスト。

torch/sam2/CUDA 不要。実 segments.npz + manifest を作り、review_index 構築・候補
プロバイダ・絞り込み/並べ替えを検証する。
"""

import cv2
import numpy as np
import pytest

from ai import amg_manifest as M, amg_npz, amg_rle
from core import current_image_amg_controller as cc


MODEL = {"model_id": "m", "sam2_commit": "c", "checkpoint_fingerprint": "f"}


def _ann(m, iou=0.9, stab=0.9):
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


def _rect(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), np.uint8); m[y0:y1, x0:x1] = 1
    return m


def _build_cache(tmp_path, key="サブ/画像 001.png"):
    h, w = 64, 64
    A = _rect(h, w, 5, 45, 5, 45)            # 1600 大
    A2 = A.copy(); A2[5, 5] = 0; A2[5, 45] = 1   # A のほぼ複製 (高IoU, 面積≈同)
    S = _rect(h, w, 10, 20, 10, 20)          # 100 A 内 (親子)
    C = _rect(h, w, 48, 60, 48, 60)          # 144 独立
    arrays = amg_npz.build_segment_arrays(
        [_ann(A, iou=0.80, stab=0.80), _ann(A2, iou=0.99, stab=0.99),
         _ann(S), _ann(C)], h, w)

    cache_dir = tmp_path / "segmentation_cache" / "images" / M.cache_id_for(key)
    cache_dir.mkdir(parents=True)
    src = tmp_path / "images" / "画像 001.png"
    src.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", np.full((h, w, 3), 100, np.uint8)); buf.tofile(str(src))
    sha = amg_npz.save_segments_npz(cache_dir / "segments.npz", arrays)
    man = M.build_image_manifest(
        image_key=key, source_path=str(src), width=w, height=h, model=MODEL,
        generator=M.preset_settings("fast"), preset="fast",
        segment_count=4, segment_ids=arrays["segment_ids"].tolist(),
        segments_npz_sha256=sha, processing_time_sec=1.0)
    M.atomic_write_json(cache_dir / "manifest.json", man)
    return cache_dir, str(src), arrays, (h, w)


# ---------------- キャッシュ状態 ----------------

def test_status_for_missing(tmp_path):
    assert cc.CurrentImageAmgController.status_for(tmp_path / "nope") == cc.STATE_MISSING


def test_status_for_ready(tmp_path):
    cache_dir, *_ = _build_cache(tmp_path)
    assert cc.CurrentImageAmgController.status_for(cache_dir) == cc.STATE_READY


def test_evaluate_reusable_then_stale(tmp_path):
    cache_dir, src, _arrays, _ = _build_cache(tmp_path)
    gen = M.preset_settings("fast")
    st = cc.CurrentImageAmgController.evaluate(
        cache_dir, source_path=src, model=MODEL, generator=gen)
    assert st.state == cc.STATE_READY
    assert st.total_candidates == 4
    # 元画像を更新 -> stale
    import os, time
    time.sleep(0.01)
    with open(src, "ab") as f:
        f.write(b"x")
    st2 = cc.CurrentImageAmgController.evaluate(
        cache_dir, source_path=src, model=MODEL, generator=gen)
    assert st2.state == cc.STATE_STALE


# ---------------- 読込と候補プロバイダ ----------------

def test_load_builds_provider_and_counts(tmp_path):
    cache_dir, *_ = _build_cache(tmp_path)
    ctl = cc.CurrentImageAmgController()
    n = ctl.load(cache_dir)
    assert n == 4
    assert ctl.is_loaded
    # 代表候補は 3 (A/A2 が 1 グループに畳まれる)
    assert ctl.representative_count() == 3


def test_parent_child_nested(tmp_path):
    cache_dir, _src, arrays, _ = _build_cache(tmp_path)
    ctl = cc.CurrentImageAmgController()
    ctl.load(cache_dir)
    areas = np.asarray(arrays["area"])
    s_idx = int(np.where(areas == 100)[0][0])
    s_sid = ctl.id_of(s_idx)
    parent_sid = ctl.parent_of(s_sid)
    assert parent_sid is not None
    # 親は面積 1600 の大候補
    assert int(areas[ctl.index_of(parent_sid)]) == 1600


def test_candidate_mask_lazy_decode(tmp_path):
    cache_dir, _src, _arrays, (h, w) = _build_cache(tmp_path)
    ctl = cc.CurrentImageAmgController()
    ctl.load(cache_dir)
    m = ctl.candidate_mask(0)
    assert m.shape == (h, w)
    assert set(np.unique(m)).issubset({0, 255})


def test_candidates_at_point(tmp_path):
    cache_dir, _src, arrays, _ = _build_cache(tmp_path)
    ctl = cc.CurrentImageAmgController()
    ctl.load(cache_dir)
    # C の中心 (54,54) は C だけ
    cands = ctl.candidates_at(54, 54)
    areas = np.asarray(arrays["area"])
    assert len(cands) == 1 and int(areas[cands[0]]) == 144


def test_visible_representatives_only(tmp_path):
    cache_dir, *_ = _build_cache(tmp_path)
    ctl = cc.CurrentImageAmgController()
    ctl.load(cache_dir)
    vis = ctl.visible_indices(representatives_only=True, hide_covered=False)
    assert len(vis) == 3
    all_vis = ctl.visible_indices(representatives_only=False, hide_covered=False)
    assert len(all_vis) == 4


def test_hide_covered_hides_nested(tmp_path):
    cache_dir, _src, arrays, _ = _build_cache(tmp_path)
    ctl = cc.CurrentImageAmgController()
    ctl.load(cache_dir)
    areas = np.asarray(arrays["area"])
    # 代表の大候補 (1600) を REMOVE 済みにする
    rep_big = next(i for i in ctl.representative_indices() if int(areas[i]) == 1600)
    s_idx = int(np.where(areas == 100)[0][0])
    vis = ctl.visible_indices(
        representatives_only=True, hide_covered=True, removed_indices=[rep_big])
    assert rep_big in vis            # REMOVE 済みは表示
    assert s_idx not in vis          # 大候補に包含された入れ子は隠れる


def test_sort_by_area_desc(tmp_path):
    cache_dir, _src, arrays, _ = _build_cache(tmp_path)
    ctl = cc.CurrentImageAmgController()
    ctl.load(cache_dir)
    vis = ctl.visible_indices(representatives_only=False, hide_covered=False, sort_mode="area")
    areas = np.asarray(arrays["area"])
    sizes = [int(areas[i]) for i in vis]
    assert sizes == sorted(sizes, reverse=True)


def test_unload_clears(tmp_path):
    cache_dir, *_ = _build_cache(tmp_path)
    ctl = cc.CurrentImageAmgController()
    ctl.load(cache_dir)
    ctl.unload()
    assert not ctl.is_loaded
    assert ctl.total_candidates == 0


def test_query_before_load_raises(tmp_path):
    ctl = cc.CurrentImageAmgController()
    with pytest.raises(RuntimeError):
        ctl.representative_indices()
