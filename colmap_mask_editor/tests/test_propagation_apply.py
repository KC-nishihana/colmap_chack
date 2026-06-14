"""一括適用 (add/exclude/replace) のトランザクションテスト (torch不要)。"""

import numpy as np

from ai.ai_mask_ops import APPLY_ADD, APPLY_EXCLUDE, APPLY_REPLACE
from ai.propagation_staging import read_mask_png, write_mask_png_atomic
from core.propagation_apply_worker import ApplyTarget, apply_batch


def _mask(h, w, box=None):
    m = np.zeros((h, w), np.uint8)
    if box:
        x0, y0, x1, y1 = box
        m[y0:y1, x0:x1] = 255
    return m


def _targets(tmp_path, results):
    out = []
    for i, (key, res) in enumerate(results):
        rp = tmp_path / "results" / f"{i:06d}.png"
        write_mask_png_atomic(rp, res)
        save = tmp_path / "masks" / f"{key}.png"
        out.append(ApplyTarget(entry_key=key, save_path=str(save), result_mask_path=str(rp)))
    return out


def test_apply_add_unions(tmp_path):
    save = tmp_path / "masks" / "e0.png"
    write_mask_png_atomic(save, _mask(40, 40, (0, 0, 10, 40)))   # 左帯 既存
    res = _mask(40, 40, (30, 0, 40, 40))                          # 右帯 結果
    tgt = ApplyTarget("e0", str(save), str(tmp_path / "r.png"))
    write_mask_png_atomic(tmp_path / "r.png", res)

    apply_batch([tgt], APPLY_ADD, tmp_path / "backup")
    out = read_mask_png(save)
    assert out[0, 5] == 255 and out[0, 35] == 255   # 両方の帯が残る
    assert out[0, 20] == 0


def test_apply_exclude_removes(tmp_path):
    save = tmp_path / "masks" / "e0.png"
    write_mask_png_atomic(save, _mask(40, 40, (0, 0, 40, 40)))   # 全面 既存
    res = _mask(40, 40, (0, 0, 20, 40))                          # 左半分 除外
    tgt = ApplyTarget("e0", str(save), str(tmp_path / "r.png"))
    write_mask_png_atomic(tmp_path / "r.png", res)

    apply_batch([tgt], APPLY_EXCLUDE, tmp_path / "backup")
    out = read_mask_png(save)
    assert out[0, 5] == 0 and out[0, 35] == 255


def test_apply_replace(tmp_path):
    save = tmp_path / "masks" / "e0.png"
    write_mask_png_atomic(save, _mask(40, 40, (0, 0, 40, 40)))
    res = _mask(40, 40, (10, 10, 20, 20))
    tgt = ApplyTarget("e0", str(save), str(tmp_path / "r.png"))
    write_mask_png_atomic(tmp_path / "r.png", res)

    apply_batch([tgt], APPLY_REPLACE, tmp_path / "backup")
    out = read_mask_png(save)
    assert int((out > 0).sum()) == int((res > 0).sum())


def test_apply_creates_new_mask_when_absent(tmp_path):
    res = _mask(32, 32, (5, 5, 15, 15))
    tgt = ApplyTarget("new", str(tmp_path / "masks" / "new.png"), str(tmp_path / "r.png"))
    write_mask_png_atomic(tmp_path / "r.png", res)

    outcome = apply_batch([tgt], APPLY_ADD, tmp_path / "backup")
    assert (tmp_path / "masks" / "new.png").exists()
    assert outcome.applied == ["new"]
    # 新規作成は record で existed=False
    assert outcome.record["targets"][0]["existed"] is False


def test_apply_writes_record_and_no_staged_leftover(tmp_path):
    tgts = _targets(tmp_path, [("a", _mask(20, 20, (0, 0, 10, 10))),
                               ("b", _mask(20, 20, (0, 0, 5, 5)))])
    outcome = apply_batch(tgts, APPLY_ADD, tmp_path / "backup", job_id="prop-x")
    assert outcome.record_path is not None
    assert outcome.record["apply_mode"] == "add"
    assert len(outcome.record["targets"]) == 2
    assert list((tmp_path / "backup" / "staged").glob("*.png")) == []


def test_apply_does_not_touch_unaccepted(tmp_path):
    # 採用していない (targets に含めない) 既存マスクは変更されない
    other = tmp_path / "masks" / "other.png"
    write_mask_png_atomic(other, _mask(20, 20, (0, 0, 20, 20)))
    before = other.read_bytes()
    tgts = _targets(tmp_path, [("a", _mask(20, 20, (0, 0, 10, 10)))])
    apply_batch(tgts, APPLY_REPLACE, tmp_path / "backup")
    assert other.read_bytes() == before
