"""画像順序・重複除去・範囲選択のテスト (torch不要)。"""

import pytest

from ai.propagation_order import (
    PropagationOrder,
    SourceImage,
    order_images,
    select_explicit,
    select_range,
)
from ai.propagation_protocol import PropagationDirection


def _imgs():
    # list順とファイル名順・colmap順・撮影時刻順がそれぞれ異なるように作る
    return [
        SourceImage("a", "p/a", list_index=0, file_name="IMG_10.jpg", colmap_index=2, capture_time=300.0),
        SourceImage("b", "p/b", list_index=1, file_name="IMG_2.jpg", colmap_index=0, capture_time=100.0),
        SourceImage("c", "p/c", list_index=2, file_name="IMG_1.jpg", colmap_index=1, capture_time=200.0),
    ]


def test_current_list_order():
    out = order_images(_imgs(), PropagationOrder.CURRENT_LIST)
    assert [i.entry_key for i in out] == ["a", "b", "c"]


def test_file_name_natural_order():
    out = order_images(_imgs(), PropagationOrder.FILE_NAME)
    # IMG_1 < IMG_2 < IMG_10 (自然順)
    assert [i.file_name for i in out] == ["IMG_1.jpg", "IMG_2.jpg", "IMG_10.jpg"]


def test_colmap_priority_order():
    out = order_images(_imgs(), PropagationOrder.COLMAP_PRIORITY)
    assert [i.entry_key for i in out] == ["b", "c", "a"]  # colmap 0,1,2


def test_capture_time_order():
    out = order_images(_imgs(), PropagationOrder.CAPTURE_TIME)
    assert [i.entry_key for i in out] == ["b", "c", "a"]  # 100,200,300


def test_colmap_priority_missing_go_last():
    imgs = [
        SourceImage("a", "p/a", 0, "a.jpg", colmap_index=None),
        SourceImage("b", "p/b", 1, "b.jpg", colmap_index=5),
    ]
    out = order_images(imgs, PropagationOrder.COLMAP_PRIORITY)
    assert [i.entry_key for i in out] == ["b", "a"]


def test_dedup_keeps_first():
    imgs = [
        SourceImage("a", "p/a", 0, "a.jpg"),
        SourceImage("a", "p/a2", 1, "a2.jpg"),
        SourceImage("b", "p/b", 2, "b.jpg"),
    ]
    out = order_images(imgs, PropagationOrder.CURRENT_LIST)
    assert [i.entry_key for i in out] == ["a", "b"]
    assert out[0].source_path == "p/a"


def _seq(n):
    return [SourceImage(f"e{i}", f"p/{i}", i, f"{i}.jpg") for i in range(n)]


def test_select_range_forward():
    frames, ref = select_range(_seq(10), "e5", PropagationDirection.FORWARD, 3)
    assert [f.entry_key for f in frames] == ["e5", "e6", "e7", "e8"]
    assert ref == 0


def test_select_range_backward():
    frames, ref = select_range(_seq(10), "e5", PropagationDirection.BACKWARD, 2)
    assert [f.entry_key for f in frames] == ["e3", "e4", "e5"]
    assert ref == 2


def test_select_range_both_clamps():
    frames, ref = select_range(_seq(5), "e1", PropagationDirection.BOTH, 10)
    assert [f.entry_key for f in frames] == ["e0", "e1", "e2", "e3", "e4"]
    assert ref == 1


def test_select_range_reference_always_included():
    frames, ref = select_range(_seq(3), "e0", PropagationDirection.BACKWARD, 5)
    assert frames[ref].entry_key == "e0"


def test_select_explicit_adds_reference_and_orders():
    frames, ref = select_explicit(_seq(10), ["e7", "e3"], "e5")
    assert [f.entry_key for f in frames] == ["e3", "e5", "e7"]
    assert frames[ref].entry_key == "e5"


def test_find_ref_missing_raises():
    with pytest.raises(ValueError):
        select_range(_seq(3), "zzz", PropagationDirection.FORWARD, 1)
