"""V0.9: 一括適用の複数画像ロールバック・バッチ取り消し。"""

import cv2
import numpy as np

from ai import partition_npz, partition_manifest as pman
from ai.amg_manifest import atomic_write_json
from core import partition_apply_worker as paw
from core.partition_apply_worker import PartitionApplyTarget

from tests._partition_helpers import simple_three_leaf


def _cache(tmp_path, name, decisions):
    cache = tmp_path / name
    cache.mkdir(parents=True)
    sha = partition_npz.save_partition_npz(cache / pman.PARTITION_NPZ_NAME, simple_three_leaf())
    atomic_write_json(cache / pman.PARTITION_REVIEW_NAME,
                      pman.build_partition_review(partition_npz_sha256=sha,
                                                  target_visible_count=2,
                                                  node_decisions=decisions))
    return cache


def test_multi_image_undo_restores_all(tmp_path):
    caches = [_cache(tmp_path, f"c{i}", {"5": "keep"}) for i in range(3)]
    saves = []
    for i in range(3):
        sp = tmp_path / f"out{i}.png"
        sp.write_bytes(cv2.imencode(".png", np.full((4, 6), 9, np.uint8))[1].tobytes())
        saves.append(sp)
    targets = [PartitionApplyTarget(f"k{i}", str(caches[i]), str(saves[i])) for i in range(3)]

    outcome = paw.apply_partition_batch(targets, tmp_path / "bk", job_id="batch")
    for sp in saves:
        m = cv2.imdecode(np.fromfile(str(sp), np.uint8), cv2.IMREAD_GRAYSCALE)
        assert np.all(m == 255)  # 全 keep

    restored = paw.undo_partition_batch(outcome.record)
    assert len(restored) == 3
    for sp in saves:
        m = cv2.imdecode(np.fromfile(str(sp), np.uint8), cv2.IMREAD_GRAYSCALE)
        assert np.all(m == 9)  # 元へロールバック
