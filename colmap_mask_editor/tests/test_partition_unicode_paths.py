"""V0.9: 日本語・全角スペースを含む Windows パスでの partition 生成と最終マスク保存。"""

import cv2
import numpy as np

from ai import partition_npz, partition_manifest as pman
from ai import partition_mask_composer as pmc
from partition_backend import partition_builder as builder

from tests._partition_helpers import synthetic_bgr, simple_three_leaf


def test_build_partition_unicode_fullwidth_space_path(tmp_path):
    # 日本語 + 全角スペース (　) を含むディレクトリ/ファイル名
    base = tmp_path / "テスト　データ" / "サブ フォルダ"
    base.mkdir(parents=True)
    img_path = base / "画像　001.png"
    img = synthetic_bgr(80, 100, seed=21)
    img_path.write_bytes(cv2.imencode(".png", img)[1].tobytes())

    out = base / "ｷｬｯｼｭ 出力"
    manifest = builder.build_partition(
        img_path, image_key="テスト/画像001", output_dir=out,
        settings={"backend": "auto", "working_max_side": 0,
                  "base_region_count": 40, "default_visible_count": 20,
                  "min_region_area_ratio": 10})
    assert manifest["coverage"]["coverage_ratio"] == 1.0
    partition_npz.verify_partition_npz(out / pman.PARTITION_NPZ_NAME)


def test_load_image_unicode():
    # load_image_unicode は np.fromfile + cv2.imdecode で全角パスを読む
    import tempfile
    from pathlib import Path
    d = Path(tempfile.mkdtemp()) / "全角　テスト"
    d.mkdir(parents=True)
    p = d / "あ い う.png"
    cv2.imencode(".png", np.zeros((5, 7, 3), np.uint8))[1].tofile(str(p))
    img = builder.load_image_unicode(p)
    assert img.shape[:2] == (5, 7)


def test_save_mask_unicode(tmp_path):
    arr = simple_three_leaf()
    lut = pmc.leaf_decision_values(arr["node_parent"], 3, {"5": "keep"})
    mask = pmc.compose_mask(arr["run_region_ids"], arr["run_lengths"], 4, 6, lut)
    path = tmp_path / "結果　マスク" / "ﾏｽｸ.png"
    pmc.save_mask_png(path, mask)
    loaded = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE)
    assert np.array_equal(loaded, mask)
