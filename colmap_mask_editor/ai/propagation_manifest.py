"""
V0.7: 伝播の frame_manifest / result_manifest の原子的な読み書き (stdlib のみ)。

JSON を一時ファイルへ書いてから os.replace で確定する。マスク本体は埋め込まない。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def atomic_write_json(path, data: dict[str, Any]) -> None:
    """JSON を tmp へ書き fsync して os.replace で確定する。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp, p)


def read_json(path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_frame_manifest(
    job_id: str,
    reference_frame_index: int,
    width: int,
    height: int,
    frames: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "reference_frame_index": reference_frame_index,
        "width": width,
        "height": height,
        "frames": frames,
    }


def build_result_manifest(
    job_id: str,
    reference_frame_index: int,
    width: int,
    height: int,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "reference_frame_index": reference_frame_index,
        "width": width,
        "height": height,
        "results": results,
    }
