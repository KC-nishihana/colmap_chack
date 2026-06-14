"""日本語・全角スペースを含むパスでのステージング+一括適用 (torch不要)。"""

import cv2
import numpy as np

from ai.ai_mask_ops import APPLY_ADD
from ai.propagation_staging import read_mask_png, stage_sequence, write_mask_png_atomic
from core.propagation_apply_worker import ApplyTarget, apply_batch, undo_batch


def _save_jpg(path, w=96, h=72):
    img = np.zeros((h, w, 3), np.uint8)
    cv2.rectangle(img, (5, 5), (w - 5, h - 5), (200, 200, 200), -1)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    buf.tofile(str(path))


def test_stage_and_apply_with_japanese_paths(tmp_path):
    src = tmp_path / "日本語 プロジェクト" / "画像"
    src.mkdir(parents=True)
    entries = []
    for i in range(3):
        p = src / f"画像 {i:03d}.jpg"
        _save_jpg(p)
        entries.append({"frame_index": i, "entry_key": f"画像 {i:03d}.jpg", "source_path": str(p)})

    frames_dir = tmp_path / "実行時 フォルダ" / "frames"
    man = stage_sequence(entries, frames_dir, reference_frame_index=1)
    assert man["width"] == 96 and man["height"] == 72
    assert (frames_dir / "000000.jpg").exists()

    # 結果PNGを日本語パスへ作り、日本語の保存先へ適用
    res = np.zeros((72, 96), np.uint8)
    res[10:30, 10:40] = 255
    rp = tmp_path / "結果 出力" / "000001.png"
    write_mask_png_atomic(rp, res)
    save = tmp_path / "マスク 出力" / "画像 001.png"
    tgt = ApplyTarget("画像 001.jpg", str(save), str(rp))

    outcome = apply_batch([tgt], APPLY_ADD, tmp_path / "バックアップ")
    assert save.exists()
    out = read_mask_png(save)
    assert int((out > 0).sum()) == int((res > 0).sum())

    # 取り消し (新規作成だったので削除)
    undo_batch(outcome.record)
    assert not save.exists()
