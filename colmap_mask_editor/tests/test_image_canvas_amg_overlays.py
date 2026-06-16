"""V0.11: ImageCanvas へ追加した AMG候補 / AIクリック オーバーレイ API のテスト。

通常 pytest (pytest-qt)。torch / sam2 / amg_rle 非依存。canvas は dense マスクを
受け取るだけで、全候補を同時保持しない。
"""

import numpy as np
import pytest

from ui.image_canvas import ImageCanvas


def _gray_image(h=24, w=32, val=100):
    return np.full((h, w, 3), val, np.uint8)


def _region(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_as_overlay_mask_normalizes():
    assert ImageCanvas._as_overlay_mask(None) is None
    b = np.zeros((4, 4), bool); b[1, 1] = True
    out = ImageCanvas._as_overlay_mask(b)
    assert out.dtype == bool and out[1, 1]
    u = np.zeros((4, 4), np.uint8); u[2, 2] = 255
    out2 = ImageCanvas._as_overlay_mask(u)
    assert out2[2, 2] and not out2[0, 0]
    # 3D は先頭チャンネルを使う
    rgb = np.zeros((4, 4, 3), np.uint8); rgb[3, 3, 0] = 200
    assert ImageCanvas._as_overlay_mask(rgb)[3, 3]


def test_setters_store_and_clear(qtbot):
    c = ImageCanvas()
    qtbot.addWidget(c)
    h, w = 24, 32
    c.set_amg_remove_union(_region(h, w, 0, 5, 0, 5))
    c.set_amg_add_union(_region(h, w, 5, 10, 5, 10))
    c.set_amg_selected_candidate(_region(h, w, 10, 15, 10, 15))
    c.set_amg_hover_candidate(_region(h, w, 0, 3, 0, 3))
    c.set_interactive_ai_preview(_region(h, w, 12, 20, 12, 20))
    assert c._has_amg_overlays()
    c.clear_ai_review_overlays()
    assert not c._has_amg_overlays()
    assert c._amg_selected_mask is None


def test_candidate_provider_stored_not_decoded(qtbot):
    c = ImageCanvas()
    qtbot.addWidget(c)
    sentinel = object()
    c.set_amg_candidates(sentinel)
    assert c.amg_candidate_provider() is sentinel  # 参照保持のみ (dense 復号しない)


def test_remove_union_blends_red(qtbot):
    c = ImageCanvas()
    qtbot.addWidget(c)
    h, w = 24, 32
    c.set_image(_gray_image(h, w, 100))
    reg = _region(h, w, 4, 12, 4, 12)
    c.set_amg_remove_union(reg)
    comp = c._build_composite(w, h)   # 等倍 (needs_scale=False)
    # 赤 (BGR R=index2) が region で増え、外では不変
    assert comp[6, 6, 2] > 120        # R が増加
    assert comp[0, 0, 2] == 100       # 外は不変


def test_selected_candidate_blends_light_blue(qtbot):
    c = ImageCanvas()
    qtbot.addWidget(c)
    h, w = 24, 32
    c.set_image(_gray_image(h, w, 100))
    reg = _region(h, w, 4, 12, 4, 12)
    c.set_amg_selected_candidate(reg)
    comp = c._build_composite(w, h)
    # 水色 = BGR (255,220,120): B(index0) が大きく増える
    assert comp[6, 6, 0] > 150
    assert comp[0, 0, 0] == 100


def test_overlay_visibility_toggle_hides_layer(qtbot):
    c = ImageCanvas()
    qtbot.addWidget(c)
    h, w = 24, 32
    c.set_image(_gray_image(h, w, 100))
    reg = _region(h, w, 4, 12, 4, 12)
    c.set_amg_remove_union(reg)
    c.set_ai_review_overlay_visible("remove", False)
    comp = c._build_composite(w, h)
    assert comp[6, 6, 2] == 100       # 非表示なので不変


def test_interactive_preview_draws_boundary(qtbot):
    c = ImageCanvas()
    qtbot.addWidget(c)
    h, w = 24, 32
    c.set_image(_gray_image(h, w, 100))
    reg = _region(h, w, 6, 16, 6, 16)
    c.set_interactive_ai_preview(reg)
    comp = c._build_composite(w, h)
    # 白境界 (255,255,255) が region 周縁に描かれ、内部中央は塗りつぶされない
    white = np.all(comp == 255, axis=2)
    assert white.any()                # 境界が存在
    assert not white[11, 11]          # 中央はフィルされない (境界のみ)


def test_overlays_default_none_do_not_affect_composite(qtbot):
    c = ImageCanvas()
    qtbot.addWidget(c)
    h, w = 24, 32
    c.set_image(_gray_image(h, w, 100))
    comp = c._build_composite(w, h)
    assert np.all(comp == 100)        # オーバーレイ未設定なら元画像のまま


def test_overlays_render_at_display_scale_8k_safe(qtbot):
    # 大画像でも _build_composite は表示解像度で処理する (full dense を毎回作らない)。
    c = ImageCanvas()
    qtbot.addWidget(c)
    h, w = 400, 600          # 縮小表示を模す原寸
    c.set_image(_gray_image(h, w, 100))
    c.set_amg_remove_union(_region(h, w, 50, 200, 50, 200))
    comp = c._build_composite(150, 100)   # 表示解像度 (縮小)
    assert comp.shape[:2] == (100, 150)   # 表示解像度で生成される
