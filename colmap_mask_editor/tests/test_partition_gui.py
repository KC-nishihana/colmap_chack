"""V0.9: GUI (パネル / レビューモデル / レビューウィジェット) の挙動と非torch検証。"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ai.partition_review_model import PartitionReviewModel
from ai.partition_review_state import RegionDecision

from tests._partition_helpers import (
    build_partition_from_labels,
    synthetic_bgr,
    simple_three_leaf,
)

PKG_ROOT = Path(__file__).resolve().parent.parent


def _model_from_image():
    from partition_backend import watershed_backend
    img = synthetic_bgr(120, 160, seed=11)
    labels = watershed_backend.grid_watershed(img, base_region_count=50)
    arrays = build_partition_from_labels(labels, img)
    return img, PartitionReviewModel(arrays)


# ---------------- レビューモデル (粒度/分割/親戻し/未確認/画素率) ---------------- #

def test_model_granularity_coarse_standard_detailed():
    _, model = _model_from_image()
    for count in (10, 20, 30):
        model.set_target_visible_count(count)
        assert len(model.visible) <= max(count, 1) + 1
        assert 1 <= len(model.visible) <= model.tree.leaf_count


def test_model_granularity_keeps_decisions():
    _, model = _model_from_image()
    model.set_target_visible_count(10)
    root = model.tree.root_id
    model.set_decision(root, RegionDecision.REMOVE)
    model.set_target_visible_count(30)  # 粒度変更
    assert model.decisions  # 判断は保持
    # 全葉が remove を継承
    assert model.pixel_rates()["remove_ratio"] == pytest.approx(1.0)


def test_model_local_split():
    _, model = _model_from_image()
    model.set_target_visible_count(5)
    n = len(model.visible)
    # 内部ノードを 1 つ split
    internal = [v for v in model.visible if not model.tree.is_leaf(v)]
    if internal:
        assert model.split(internal[0])
        assert len(model.visible) == n + 1


def test_model_collapse_to_parent():
    arrays = simple_three_leaf()
    model = PartitionReviewModel(arrays)
    model.visible = [1, 2, 3]
    parent = model.collapse(1)
    assert parent == 4
    assert set(model.visible) == {4, 3}


def test_model_next_unreviewed():
    arrays = simple_three_leaf()
    model = PartitionReviewModel(arrays)
    model.visible = [1, 2, 3]
    first = model.next_unreviewed(None)
    assert first in (1, 2, 3)
    model.set_decision(first, RegionDecision.KEEP)
    nxt = model.next_unreviewed(first)
    assert nxt != first and model.effective(nxt) == "unreviewed"


def test_model_pixel_rates_and_stats():
    arrays = simple_three_leaf()
    model = PartitionReviewModel(arrays)
    model.visible = [1, 2, 3]
    model.set_decision(1, RegionDecision.KEEP)
    model.set_decision(2, RegionDecision.REMOVE)
    rates = model.pixel_rates()
    assert rates["keep_ratio"] == pytest.approx(8 / 24)
    assert rates["remove_ratio"] == pytest.approx(8 / 24)
    assert rates["unreviewed_ratio"] == pytest.approx(8 / 24)
    stats = model.stats()
    assert stats["reviewed_visible"] == 2
    assert stats["visible_count"] == 3


def test_model_bulk_descendants_decision():
    arrays = simple_three_leaf()
    model = PartitionReviewModel(arrays)
    model.set_descendants_decision(4, RegionDecision.REMOVE)  # node4 = leaf1+leaf2
    assert model.effective(1) == "remove"
    assert model.effective(2) == "remove"


# ---------------- パネル / ウィジェット (qtbot) ---------------- #

def test_panel_options(qtbot):
    from ui.partition_panel import PartitionPanel
    panel = PartitionPanel()
    qtbot.addWidget(panel)
    opts = panel.options()
    assert opts["preset"] == "coarse"
    assert opts["default_visible_count"] == 30
    assert opts["base_region_count"] == 800
    assert opts["backend"] == "auto"


def test_review_widget_construct_and_render(qtbot):
    from ui.partition_review_widget import PartitionReviewWidget
    img, model = _model_from_image()
    w = PartitionReviewWidget()
    qtbot.addWidget(w)
    w.resize(400, 300)
    model.set_target_visible_count(20)
    w.set_partition(img, model)
    # レンダリングが pixmap を生成する
    assert w.pixmap() is not None and not w.pixmap().isNull()


def test_gui_modules_do_not_import_torch_sam2():
    code = (
        "import sys;"
        "import ui.partition_panel, ui.partition_review_widget,"
        " ui.partition_controller, ai.partition_review_model;"
        "bad=[x for x in ('torch','sam2','sam2._C') if x in sys.modules];"
        "print('OK' if not bad else 'BAD', bad)"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PKG_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["QT_QPA_PLATFORM"] = "offscreen"
    proc = subprocess.run([sys.executable, "-c", code], env=env,
                          capture_output=True, text=True, timeout=120)
    assert "OK" in proc.stdout, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
