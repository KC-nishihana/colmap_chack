"""V0.8: 最終マスク合成 (3 方式・REMOVE 優先・未確認無視) のテスト。"""

import numpy as np

from ai import amg_mask_composer as mc
from ai import amg_npz, amg_rle


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


def _make_npz():
    h, w = 10, 10
    a = np.zeros((h, w), np.uint8); a[0:6, 0:6] = 1     # seg id 1 (大)
    b = np.zeros((h, w), np.uint8); b[3:9, 3:9] = 1     # seg id 2 (中, a と重複)
    c = np.zeros((h, w), np.uint8); c[8:10, 0:2] = 1    # seg id 3 (小)
    arrays = amg_npz.build_segment_arrays([_ann(a), _ann(b), _ann(c)], h, w)
    # build は area 降順ソート: a(36) > b(36)... 同 area の場合 bbox。明示的に id を取得
    return arrays, (h, w)


def _ids_by_region(arrays):
    """テスト判定しやすいよう、各 segment の bbox を返す。"""
    return {int(sid): arrays["bbox_xywh"][i].tolist()
            for i, sid in enumerate(arrays["segment_ids"].tolist())}


def test_keep_only():
    arrays, (h, w) = _make_npz()
    sids = arrays["segment_ids"].tolist()
    decisions = {str(sids[0]): "keep"}
    decisions = {str(s): "unreviewed" for s in sids}
    decisions[str(sids[0])] = "keep"
    out = mc.compose_final_mask(arrays, decisions, mc.MODE_KEEP_ONLY)
    assert out.dtype == np.uint8
    assert out.max() == 255 and out.min() == 0
    # keep セグメントの領域だけ 255
    keep_counts = amg_rle.unpack_counts(arrays, 0)
    keep_mask = amg_rle.decode_rle(keep_counts, h, w) > 0
    assert np.array_equal(out > 0, keep_mask)


def test_exclude_remove_default_full():
    arrays, (h, w) = _make_npz()
    sids = arrays["segment_ids"].tolist()
    decisions = {str(s): "unreviewed" for s in sids}
    decisions[str(sids[1])] = "remove"
    out = mc.compose_final_mask(arrays, decisions, mc.MODE_EXCLUDE_REMOVE)
    rm_counts = amg_rle.unpack_counts(arrays, 1)
    rm_mask = amg_rle.decode_rle(rm_counts, h, w) > 0
    # remove 部分は 0, それ以外は 255
    assert np.all(out[rm_mask] == 0)
    assert np.all(out[~rm_mask] == 255)


def test_remove_wins_over_keep():
    arrays, (h, w) = _make_npz()
    sids = arrays["segment_ids"].tolist()
    # seg0 を keep, seg1 を remove。重複領域は remove が勝つ
    decisions = {str(s): "unreviewed" for s in sids}
    decisions[str(sids[0])] = "keep"
    decisions[str(sids[1])] = "remove"
    out = mc.compose_final_mask(arrays, decisions, mc.MODE_ADD_REMOVE)
    keep_mask = amg_rle.decode_rle(amg_rle.unpack_counts(arrays, 0), h, w) > 0
    rm_mask = amg_rle.decode_rle(amg_rle.unpack_counts(arrays, 1), h, w) > 0
    overlap = keep_mask & rm_mask
    assert overlap.any()  # 重複が存在する前提
    assert np.all(out[overlap] == 0)  # REMOVE 優先


def test_unreviewed_not_applied():
    arrays, (h, w) = _make_npz()
    sids = arrays["segment_ids"].tolist()
    decisions = {str(s): "unreviewed" for s in sids}
    out = mc.compose_final_mask(arrays, decisions, mc.MODE_KEEP_ONLY)
    assert out.sum() == 0  # keep が無ければ全 0


def test_add_remove_with_existing():
    arrays, (h, w) = _make_npz()
    sids = arrays["segment_ids"].tolist()
    existing = np.zeros((h, w), np.uint8); existing[0:2, 8:10] = 255
    decisions = {str(s): "unreviewed" for s in sids}
    decisions[str(sids[2])] = "keep"
    out = mc.compose_final_mask(arrays, decisions, mc.MODE_ADD_REMOVE, existing_mask=existing)
    # 既存の 255 領域は残る
    assert np.all(out[0:2, 8:10] == 255)
    keep_mask = amg_rle.decode_rle(amg_rle.unpack_counts(arrays, 2), h, w) > 0
    assert np.all(out[keep_mask] == 255)
