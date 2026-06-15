"""V0.9: 最終マスク一括適用 (KEEP=255/REMOVE=0, 未確認解消, ロールバック, 取り消し)。"""

import cv2
import numpy as np
import pytest

from ai import partition_npz
from ai import partition_manifest as pman
from core import partition_apply_worker as paw
from core.partition_apply_worker import PartitionApplyTarget, PartitionApplyError

from tests._partition_helpers import simple_three_leaf


def _cache_with_review(tmp_path, decisions, target_visible=2):
    cache = tmp_path / "cache"
    cache.mkdir(parents=True)
    sha = partition_npz.save_partition_npz(cache / pman.PARTITION_NPZ_NAME,
                                           simple_three_leaf())
    review = pman.build_partition_review(
        partition_npz_sha256=sha, target_visible_count=target_visible,
        node_decisions=decisions)
    from ai.amg_manifest import atomic_write_json
    atomic_write_json(cache / pman.PARTITION_REVIEW_NAME, review)
    return cache


def test_compose_target_mask_keep_remove(tmp_path):
    cache = _cache_with_review(tmp_path, {"1": "keep", "2": "remove", "3": "keep"})
    save = tmp_path / "mask.png"
    cv2.imencode(".png", np.zeros((4, 6), np.uint8))  # placeholder
    t = PartitionApplyTarget("k", str(cache), str(save))
    mask = paw.compose_target_mask(t)
    assert mask.shape == (4, 6)
    assert np.all(mask[:, 0:2] == 255)
    assert np.all(mask[:, 2:4] == 0)
    assert np.all(mask[:, 4:6] == 255)


def test_unreviewed_blocks_apply(tmp_path):
    cache = _cache_with_review(tmp_path, {"1": "keep"})  # 2,3 未確認
    t = PartitionApplyTarget("k", str(cache), str(tmp_path / "m.png"))
    with pytest.raises(PartitionApplyError):
        paw.compose_target_mask(t, unreviewed_action="ask")


def test_unreviewed_action_remove(tmp_path):
    cache = _cache_with_review(tmp_path, {"1": "keep"})
    t = PartitionApplyTarget("k", str(cache), str(tmp_path / "m.png"))
    mask = paw.compose_target_mask(t, unreviewed_action="remove")
    assert np.all(mask[:, 0:2] == 255)
    assert np.all(mask[:, 2:6] == 0)


def test_apply_batch_and_undo(tmp_path):
    cache = _cache_with_review(tmp_path, {"5": "keep", "3": "remove"})
    save = tmp_path / "out" / "mask.png"
    save.parent.mkdir()
    # 既存マスク (全 0)
    save.write_bytes(cv2.imencode(".png", np.zeros((4, 6), np.uint8))[1].tobytes())
    backup = tmp_path / "backup"

    t = PartitionApplyTarget("k", str(cache), str(save))
    outcome = paw.apply_partition_batch([t], backup, job_id="j1")
    written = cv2.imdecode(np.fromfile(str(save), np.uint8), cv2.IMREAD_GRAYSCALE)
    # leaf1,2 keep=255, leaf3 remove=0
    assert np.all(written[:, 0:4] == 255)
    assert np.all(written[:, 4:6] == 0)

    # 取り消し -> 元の全 0 に戻る
    restored = paw.undo_partition_batch(outcome.record)
    assert len(restored) == 1
    back = cv2.imdecode(np.fromfile(str(save), np.uint8), cv2.IMREAD_GRAYSCALE)
    assert np.all(back == 0)


def test_apply_rollback_on_failure(tmp_path):
    # 1 枚目は正常、2 枚目は未確認 (ask) で失敗 -> 全体ロールバック
    good = _cache_with_review(tmp_path / "a", {"5": "keep"})
    bad = _cache_with_review(tmp_path / "b", {"1": "keep"})  # 未確認残り
    save1 = tmp_path / "out1.png"
    save2 = tmp_path / "out2.png"
    orig = cv2.imencode(".png", np.full((4, 6), 7, np.uint8))[1].tobytes()
    save1.write_bytes(orig)
    save2.write_bytes(orig)
    backup = tmp_path / "bk"
    targets = [
        PartitionApplyTarget("a", str(good), str(save1)),
        PartitionApplyTarget("b", str(bad), str(save2)),
    ]
    with pytest.raises(PartitionApplyError):
        paw.apply_partition_batch(targets, backup, unreviewed_action="ask")
    # save1 は元のまま (ロールバック)
    m1 = cv2.imdecode(np.fromfile(str(save1), np.uint8), cv2.IMREAD_GRAYSCALE)
    assert np.all(m1 == 7)
