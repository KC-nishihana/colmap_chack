"""V0.8: クリック候補判定・Tab切替・復号LRUキャッシュのテスト (torch 不要)。"""

import numpy as np

from ai import amg_hit_test as H
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


def _overlapping_npz():
    h, w = 20, 20
    big = np.zeros((h, w), np.uint8); big[2:18, 2:18] = 1     # 面積大
    small = np.zeros((h, w), np.uint8); small[8:12, 8:12] = 1  # 面積小, big と重複
    return amg_npz.build_segment_arrays([_ann(big), _ann(small)], h, w), (h, w)


def test_candidates_smallest_first():
    arrays, (h, w) = _overlapping_npz()
    # (10,10) は big と small の両方に含まれる -> 小さい方が先頭
    cands = H.candidates_at_point(arrays, 10, 10)
    assert len(cands) == 2
    areas = [int(arrays["area"][i]) for i in cands]
    assert areas[0] < areas[1]  # 面積昇順


def test_candidates_bbox_only_region():
    arrays, (h, w) = _overlapping_npz()
    # (4,4) は big のみ (small の bbox 外)
    cands = H.candidates_at_point(arrays, 4, 4)
    assert len(cands) == 1


def test_candidates_out_of_range_and_background():
    arrays, (h, w) = _overlapping_npz()
    assert H.candidates_at_point(arrays, -1, 0) == []
    assert H.candidates_at_point(arrays, 0, 0) == []  # 背景


def test_cycle_index():
    cands = [3, 7, 9]
    assert H.cycle_index(cands, None) == 3
    assert H.cycle_index(cands, None, forward=False) == 9
    assert H.cycle_index(cands, 3) == 7
    assert H.cycle_index(cands, 9) == 3       # wrap
    assert H.cycle_index(cands, 7, forward=False) == 3
    assert H.cycle_index([], 1) is None
    assert H.cycle_index(cands, 999) == 3     # current が候補外


def test_decode_cache_lru():
    arrays, (h, w) = _overlapping_npz()
    cache = H.MaskDecodeCache(arrays, max_size=1)
    m0 = cache.get(0)
    assert m0.shape == (h, w) and m0.dtype == np.uint8
    assert len(cache) == 1
    cache.get(1)
    assert len(cache) == 1  # max_size=1 -> 古いものが追い出される
    # union は必要分だけ復号
    u = cache.union([0, 1])
    assert u.dtype == bool and u.shape == (h, w)


def test_decode_cache_matches_direct():
    arrays, (h, w) = _overlapping_npz()
    cache = H.MaskDecodeCache(arrays, max_size=8)
    for i in range(int(arrays["segment_ids"].shape[0])):
        direct = amg_rle.decode_rle(amg_rle.unpack_counts(arrays, i), h, w)
        assert np.array_equal(cache.get(i), direct)
