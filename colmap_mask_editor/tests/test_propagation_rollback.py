"""一括適用の途中失敗ロールバックテスト (torch不要)。"""

import numpy as np
import pytest

import core.propagation_apply_worker as apply_mod
from ai.ai_mask_ops import APPLY_REPLACE
from ai.propagation_staging import read_mask_png, write_mask_png_atomic
from core.propagation_apply_worker import ApplyError, ApplyTarget, apply_batch


def _mask(box):
    m = np.zeros((30, 30), np.uint8)
    x0, y0, x1, y1 = box
    m[y0:y1, x0:x1] = 255
    return m


def _make_targets(tmp_path, specs):
    tgts = []
    for i, (key, existing, result) in enumerate(specs):
        save = tmp_path / "masks" / f"{key}.png"
        if existing is not None:
            write_mask_png_atomic(save, existing)
        rp = tmp_path / "results" / f"{i}.png"
        write_mask_png_atomic(rp, result)
        tgts.append(ApplyTarget(key, str(save), str(rp)))
    return tgts


def test_rollback_restores_existing_and_removes_new(tmp_path, monkeypatch):
    specs = [
        ("e0", _mask((0, 0, 30, 30)), _mask((0, 0, 10, 10))),   # 既存あり
        ("e1", None, _mask((0, 0, 10, 10))),                     # 新規
        ("e2", _mask((0, 0, 15, 15)), _mask((0, 0, 5, 5))),      # 既存あり (コミット失敗を起こす)
    ]
    tgts = _make_targets(tmp_path, specs)
    e0_before = (tmp_path / "masks" / "e0.png").read_bytes()

    # 3つ目の os.replace で失敗させる
    real_replace = apply_mod.os.replace
    calls = {"n": 0}

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError("boom")
        return real_replace(src, dst)

    monkeypatch.setattr(apply_mod.os, "replace", flaky_replace)

    with pytest.raises(ApplyError):
        apply_batch(tgts, APPLY_REPLACE, tmp_path / "backup")

    # e0 は元のバイト列へ復元
    assert (tmp_path / "masks" / "e0.png").read_bytes() == e0_before
    # e1 は新規作成だったので削除されている
    assert not (tmp_path / "masks" / "e1.png").exists()
    # staged の後始末
    assert list((tmp_path / "backup" / "staged").glob("*.png")) == []


def test_existing_mask_unchanged_after_rollback_content(tmp_path, monkeypatch):
    specs = [
        ("e0", _mask((0, 0, 30, 30)), _mask((0, 0, 3, 3))),
        ("e1", _mask((0, 0, 20, 20)), _mask((0, 0, 3, 3))),
    ]
    tgts = _make_targets(tmp_path, specs)
    e0 = read_mask_png(tmp_path / "masks" / "e0.png")

    real_replace = apply_mod.os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("fail second")
        return real_replace(src, dst)

    monkeypatch.setattr(apply_mod.os, "replace", flaky)
    with pytest.raises(ApplyError):
        apply_batch(tgts, APPLY_REPLACE, tmp_path / "backup")

    assert np.array_equal(read_mask_png(tmp_path / "masks" / "e0.png"), e0)
