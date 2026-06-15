"""V0.10: REMOVE_ONLY 中核ロジック (基準マスク・最終マスク・画素率・covered) のテスト。"""

import numpy as np
import pytest

from ai import amg_npz, amg_rle, amg_remove_only as ro
from ai.amg_review_state import SegmentDecision


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


def _rect(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), np.uint8); m[y0:y1, x0:x1] = 1
    return m


def _npz():
    h, w = 20, 20
    a = _rect(h, w, 0, 8, 0, 8)
    b = _rect(h, w, 4, 12, 4, 12)   # a と重複
    c = _rect(h, w, 15, 20, 15, 20)
    arrays = amg_npz.build_segment_arrays([_ann(a), _ann(b), _ann(c)], h, w)
    return arrays, (h, w)


def _ids(arrays):
    return arrays["segment_ids"].tolist()


def _sid_for_rect(arrays, y0, x0):
    """bbox 左上が (x0,y0) の segment_id を返す。"""
    for i, sid in enumerate(arrays["segment_ids"].tolist()):
        bx, by = int(arrays["bbox_xywh"][i][0]), int(arrays["bbox_xywh"][i][1])
        if bx == x0 and by == y0:
            return int(sid)
    raise AssertionError("not found")


# ---------- 最終マスク ----------

def test_no_selection_full_255():
    arrays, (h, w) = _npz()
    decisions = {}
    out = ro.compose_remove_only_final(arrays, decisions, base_mode=ro.BASE_FULL)
    assert out.shape == (h, w)
    assert out.dtype == np.uint8
    assert np.all(out == 255)


def test_existing_mask_preserved():
    arrays, (h, w) = _npz()
    existing = np.zeros((h, w), np.uint8)
    existing[0:5, 0:5] = 255       # 一部だけ有効
    out = ro.compose_remove_only_final(
        arrays, {}, existing_mask=existing, base_mode=ro.BASE_EXISTING_OR_FULL)
    # REMOVE 無し -> 既存マスクと一致
    assert np.array_equal(out > 0, existing > 0)


def test_only_remove_zeroed():
    arrays, (h, w) = _npz()
    sid = _sid_for_rect(arrays, 0, 0)
    decisions = {str(sid): "remove"}
    out = ro.compose_remove_only_final(arrays, decisions, base_mode=ro.BASE_FULL)
    idx = arrays["segment_ids"].tolist().index(sid)
    rm = amg_rle.decode_rle(amg_rle.unpack_counts(arrays, idx), h, w) > 0
    assert np.all(out[rm] == 0)
    assert np.all(out[~rm] == 255)


def test_multiple_remove_union():
    arrays, (h, w) = _npz()
    s1 = _sid_for_rect(arrays, 0, 0)
    s2 = _sid_for_rect(arrays, 15, 15)
    decisions = {str(s1): "remove", str(s2): "remove"}
    out = ro.compose_remove_only_final(arrays, decisions, base_mode=ro.BASE_FULL)
    i1 = arrays["segment_ids"].tolist().index(s1)
    i2 = arrays["segment_ids"].tolist().index(s2)
    rm = ((amg_rle.decode_rle(amg_rle.unpack_counts(arrays, i1), h, w) > 0)
          | (amg_rle.decode_rle(amg_rle.unpack_counts(arrays, i2), h, w) > 0))
    assert np.all(out[rm] == 0)
    assert np.all(out[~rm] == 255)


def test_overlapping_remove_counted_once():
    arrays, (h, w) = _npz()
    s1 = _sid_for_rect(arrays, 0, 0)
    s2 = _sid_for_rect(arrays, 4, 4)   # a と重複
    decisions = {str(s1): "remove", str(s2): "remove"}
    out = ro.compose_remove_only_final(arrays, decisions, base_mode=ro.BASE_FULL)
    i1 = arrays["segment_ids"].tolist().index(s1)
    i2 = arrays["segment_ids"].tolist().index(s2)
    union = ((amg_rle.decode_rle(amg_rle.unpack_counts(arrays, i1), h, w) > 0)
             | (amg_rle.decode_rle(amg_rle.unpack_counts(arrays, i2), h, w) > 0))
    excluded = int((out == 0).sum())
    assert excluded == int(union.sum())   # 重複しても和集合 1 回ぶん


def test_unreviewed_not_applied():
    arrays, (h, w) = _npz()
    decisions = {str(s): "unreviewed" for s in _ids(arrays)}
    out = ro.compose_remove_only_final(arrays, decisions, base_mode=ro.BASE_FULL)
    assert np.all(out == 255)   # 未確認は除外しない


def test_keep_does_not_affect_output():
    arrays, (h, w) = _npz()
    decisions = {str(s): "keep" for s in _ids(arrays)}
    out = ro.compose_remove_only_final(arrays, decisions, base_mode=ro.BASE_FULL)
    assert np.all(out == 255)   # KEEP は REMOVE_ONLY 出力に影響しない


def test_existing_mask_size_mismatch_rejected():
    arrays, (h, w) = _npz()
    bad = np.zeros((h + 3, w), np.uint8)
    with pytest.raises(ro.BaseMaskSizeMismatch):
        ro.compose_remove_only_final(
            arrays, {}, existing_mask=bad, base_mode=ro.BASE_EXISTING_OR_FULL)


def test_output_is_uint8_binary():
    arrays, (h, w) = _npz()
    sid = _sid_for_rect(arrays, 0, 0)
    out = ro.compose_remove_only_final(arrays, {str(sid): "remove"}, base_mode=ro.BASE_FULL)
    assert out.dtype == np.uint8
    assert set(np.unique(out).tolist()).issubset({0, 255})


def test_output_size_matches_image():
    arrays, (h, w) = _npz()
    out = ro.compose_remove_only_final(arrays, {}, base_mode=ro.BASE_FULL)
    assert out.shape == (h, w)


# ---------- 基準マスク ----------

def test_resolve_base_existing_or_full_no_mask():
    base = ro.resolve_base_mask(10, 10, None, ro.BASE_EXISTING_OR_FULL)
    assert base.shape == (10, 10) and base.all()


def test_resolve_base_full_ignores_existing():
    existing = np.zeros((10, 10), np.uint8); existing[0:3, 0:3] = 255
    assert ro.resolve_base_existing(10, 10, existing, ro.BASE_FULL) is None


# ---------- 画素率 ----------

def test_pixel_stats():
    base = np.ones((10, 10), bool)
    remove = np.zeros((10, 10), bool); remove[0:2, 0:5] = True   # 10 px
    st = ro.pixel_stats(base, remove)
    assert st.total_px == 100
    assert st.excluded_px == 10
    assert st.effective_px == 90
    assert abs(st.excluded_ratio - 0.10) < 1e-9
    assert abs(st.effective_ratio - 0.90) < 1e-9


def test_pixel_stats_base_outside_counts_as_excluded():
    base = np.zeros((10, 10), bool); base[0:5, :] = True   # 50 px 有効
    remove = np.zeros((10, 10), bool)
    st = ro.pixel_stats(base, remove)
    assert st.effective_px == 50
    assert st.excluded_px == 50    # 基準マスク外も除外扱い


# ---------- covered 抑制 ----------

def test_covered_above_threshold():
    seg = np.zeros((10, 10), bool); seg[0:4, 0:4] = True
    remove = np.zeros((10, 10), bool); remove[0:4, 0:4] = True  # 完全包含
    assert ro.covered_ratio(seg, remove) == 1.0
    assert ro.is_covered(seg, remove, 0.98)


def test_covered_below_threshold():
    seg = np.zeros((10, 10), bool); seg[0:4, 0:4] = True   # 16 px
    remove = np.zeros((10, 10), bool); remove[0:2, 0:4] = True  # 8 px -> 50%
    assert ro.covered_ratio(seg, remove) == 0.5
    assert not ro.is_covered(seg, remove, 0.98)


def test_covered_does_not_mutate_decisions():
    decisions = {"5": "remove"}
    seg = np.zeros((10, 10), bool); seg[0:4, 0:4] = True
    remove = np.ones((10, 10), bool)
    ro.is_covered(seg, remove)
    assert decisions == {"5": "remove"}   # 純粋関数: 判断を書き換えない


# ---------- decisions 最小化 ----------

def test_prune_keeps_only_remove():
    decisions = {"1": "remove", "2": "unreviewed", "3": "keep", "4": "remove"}
    pruned = ro.prune_remove_only_decisions(decisions)
    assert pruned == {"1": "remove", "4": "remove"}


def test_remove_segment_ids_sorted():
    decisions = {"10": "remove", "2": "remove", "5": "keep"}
    assert ro.remove_segment_ids(decisions) == [2, 10]
    assert ro.count_remove(decisions) == 2
