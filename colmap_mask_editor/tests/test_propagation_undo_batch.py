"""一括適用バッチの取り消しテスト (torch不要)。"""

import numpy as np

from ai.ai_mask_ops import APPLY_REPLACE
from ai.propagation_staging import read_mask_png, write_mask_png_atomic
from core.propagation_apply_worker import ApplyTarget, apply_batch, undo_batch


def _mask(box):
    m = np.zeros((30, 30), np.uint8)
    x0, y0, x1, y1 = box
    m[y0:y1, x0:x1] = 255
    return m


def _setup(tmp_path):
    save0 = tmp_path / "masks" / "e0.png"      # 既存
    write_mask_png_atomic(save0, _mask((0, 0, 30, 30)))
    r0 = tmp_path / "results" / "0.png"
    write_mask_png_atomic(r0, _mask((0, 0, 5, 5)))
    r1 = tmp_path / "results" / "1.png"
    write_mask_png_atomic(r1, _mask((0, 0, 7, 7)))
    save1 = tmp_path / "masks" / "e1.png"      # 新規
    return [
        ApplyTarget("e0", str(save0), str(r0)),
        ApplyTarget("e1", str(save1), str(r1)),
    ], save0, save1


def test_undo_restores_existing_and_deletes_new(tmp_path):
    tgts, save0, save1 = _setup(tmp_path)
    e0_before = read_mask_png(save0)

    outcome = apply_batch(tgts, APPLY_REPLACE, tmp_path / "backup")
    # 適用後は変わっている
    assert not np.array_equal(read_mask_png(save0), e0_before)
    assert save1.exists()

    undone = undo_batch(outcome.record)
    assert set(undone) == {"e0", "e1"}
    # e0 は元へ復元
    assert np.array_equal(read_mask_png(save0), e0_before)
    # e1 (新規) は削除
    assert not save1.exists()


def test_undo_from_record_path(tmp_path):
    tgts, save0, save1 = _setup(tmp_path)
    e0_before = read_mask_png(save0)
    outcome = apply_batch(tgts, APPLY_REPLACE, tmp_path / "backup")
    undone = undo_batch(outcome.record_path)   # JSONパスからも取り消せる
    assert set(undone) == {"e0", "e1"}
    assert np.array_equal(read_mask_png(save0), e0_before)
