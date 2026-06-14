"""フレームステージングのテスト (cv2 必要・torch不要)。日本語パス対応。"""

import cv2
import numpy as np
import pytest

from ai.propagation_staging import (
    StagingError,
    read_mask_png,
    stage_sequence,
    write_mask_png_atomic,
)


def _save_jpg(path, w, h):
    img = np.zeros((h, w, 3), np.uint8)
    cv2.rectangle(img, (5, 5), (w - 5, h - 5), (200, 200, 200), -1)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    buf.tofile(str(path))


def test_stage_sequence_numbered_and_dims(tmp_path):
    src = tmp_path / "日本語 画像"  # 日本語+全角スペース
    src.mkdir()
    entries = []
    for i in range(3):
        p = src / f"画像 {i}.jpg"
        _save_jpg(p, 128, 96)
        entries.append({"frame_index": i, "entry_key": f"画像 {i}.jpg", "source_path": str(p)})

    out = tmp_path / "out" / "frames"
    man = stage_sequence(entries, out, reference_frame_index=1, jpeg_quality=95)

    assert man["width"] == 128 and man["height"] == 96
    assert man["reference_frame_index"] == 1
    assert [f["frame_index"] for f in man["frames"]] == [0, 1, 2]
    # 連番ゼロ埋めJPEGが存在する
    for i in range(3):
        assert (out / f"{i:06d}.jpg").exists()
    # 一時ファイルが残っていない
    assert list(out.glob("*.tmp")) == []


def test_stage_sequence_rejects_size_mismatch(tmp_path):
    src = tmp_path / "s"
    src.mkdir()
    entries = []
    sizes = [(128, 96), (64, 64)]
    for i, (w, h) in enumerate(sizes):
        p = src / f"{i}.jpg"
        _save_jpg(p, w, h)
        entries.append({"frame_index": i, "entry_key": f"{i}", "source_path": str(p)})
    with pytest.raises(StagingError):
        stage_sequence(entries, tmp_path / "out", reference_frame_index=0)


def test_stage_sequence_requires_contiguous_index(tmp_path):
    src = tmp_path / "s"
    src.mkdir()
    p = src / "x.jpg"
    _save_jpg(p, 32, 32)
    entries = [{"frame_index": 3, "entry_key": "x", "source_path": str(p)}]
    with pytest.raises(StagingError):
        stage_sequence(entries, tmp_path / "out", reference_frame_index=3)


def test_mask_png_roundtrip_atomic(tmp_path):
    m = np.zeros((40, 60), np.uint8)
    m[5:15, 5:25] = 255
    dest = tmp_path / "結果" / "000001.png"
    write_mask_png_atomic(dest, m)
    assert dest.exists()
    assert list(dest.parent.glob("*.tmp")) == []
    back = read_mask_png(dest)
    assert back.dtype == np.uint8
    assert set(np.unique(back)).issubset({0, 255})
    assert int((back > 0).sum()) == int((m > 0).sum())
