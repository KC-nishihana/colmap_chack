"""V0.10: 重複候補グループ化と代表候補選択のテスト。"""

import numpy as np

from ai import amg_candidate_grouping as gp
from ai import amg_npz, amg_rle


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


def _rect(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), np.uint8); m[y0:y1, x0:x1] = 1
    return m


def _sid_to_index(arrays):
    return {int(s): i for i, s in enumerate(arrays["segment_ids"].tolist())}


def test_high_iou_same_group():
    h, w = 50, 50
    a = _rect(h, w, 5, 35, 5, 35)   # 30x30
    b = _rect(h, w, 6, 36, 6, 36)   # 1px ずれ -> IoU ~0.88
    arrays = amg_npz.build_segment_arrays([_ann(a), _ann(b)], h, w)
    res = gp.group_candidates(arrays, iou_threshold=0.85, containment_threshold=0.95)
    assert res.group_count == 1
    assert len(set(res.group_ids.tolist())) == 1


def test_containment_same_group_low_iou():
    h, w = 60, 60
    outer = _rect(h, w, 0, 50, 0, 50)        # area 2500
    inner = _rect(h, w, 5, 15, 5, 15)        # area 100, 完全に outer 内 -> containment 1.0, IoU 低
    arrays = amg_npz.build_segment_arrays([_ann(outer), _ann(inner)], h, w)
    res = gp.group_candidates(arrays, iou_threshold=0.85, containment_threshold=0.95)
    assert res.group_count == 1


def test_distinct_objects_different_groups():
    h, w = 40, 40
    a = _rect(h, w, 0, 10, 0, 10)
    b = _rect(h, w, 30, 40, 30, 40)
    arrays = amg_npz.build_segment_arrays([_ann(a), _ann(b)], h, w)
    res = gp.group_candidates(arrays)
    assert res.group_count == 2
    assert len(set(res.group_ids.tolist())) == 2


def test_bbox_non_intersect_not_compared():
    # bbox が交差しなければ RLE 比較自体を行わない -> 別グループ
    h, w = 50, 50
    a = _rect(h, w, 0, 20, 0, 20)
    b = _rect(h, w, 25, 45, 25, 45)
    arrays = amg_npz.build_segment_arrays([_ann(a), _ann(b)], h, w)
    assert not gp.bbox_intersects(arrays["bbox_xywh"][0], arrays["bbox_xywh"][1])
    res = gp.group_candidates(arrays)
    assert res.group_count == 2


def test_representative_deterministic_by_quality():
    h, w = 50, 50
    a = _rect(h, w, 5, 35, 5, 35)
    b = _rect(h, w, 6, 36, 6, 36)
    # b の quality を高くする -> b が代表
    arrays = amg_npz.build_segment_arrays(
        [_ann(a, iou=0.80, stab=0.90), _ann(b, iou=0.99, stab=0.99)], h, w)
    res = gp.group_candidates(arrays)
    s2i = _sid_to_index(arrays)
    quality = gp.quality_scores(arrays)
    rep_sid = int(res.representative_segment_ids[0])
    rep_idx = s2i[rep_sid]
    # 代表が最高 quality を持つ
    assert quality[rep_idx] == max(quality)
    assert res.is_representative[rep_idx] == 1


def test_same_input_same_group_id():
    h, w = 50, 50
    masks = [_rect(h, w, 0, 10, 0, 10), _rect(h, w, 1, 11, 1, 11),
             _rect(h, w, 40, 50, 40, 50)]
    arrays = amg_npz.build_segment_arrays([_ann(m) for m in masks], h, w)
    r1 = gp.group_candidates(arrays)
    r2 = gp.group_candidates(arrays)
    assert np.array_equal(r1.group_ids, r2.group_ids)
    assert np.array_equal(r1.representative_segment_ids, r2.representative_segment_ids)


def test_threshold_change_changes_grouping():
    h, w = 60, 60
    outer = _rect(h, w, 0, 50, 0, 50)
    inner = _rect(h, w, 5, 15, 5, 15)        # containment 1.0
    arrays = amg_npz.build_segment_arrays([_ann(outer), _ann(inner)], h, w)
    # containment しきい値を 1.0 超に上げ、IoU も高くすれば別グループになる
    res = gp.group_candidates(arrays, iou_threshold=0.99, containment_threshold=1.01)
    assert res.group_count == 2


def test_edge_touch_flags():
    h, w = 40, 40
    edge = _rect(h, w, 0, 10, 0, 10)         # 端に接する
    center = _rect(h, w, 15, 25, 15, 25)     # 中央
    arrays = amg_npz.build_segment_arrays([_ann(edge), _ann(center)], h, w)
    s2i = _sid_to_index(arrays)
    flags = gp.edge_touch_flags(arrays)
    # edge の index は bbox が (0,0,...) のもの
    edge_idx = next(i for i in range(2) if list(arrays["bbox_xywh"][i][:2]) == [0, 0])
    center_idx = 1 - edge_idx
    assert flags[edge_idx] == 1
    assert flags[center_idx] == 0
