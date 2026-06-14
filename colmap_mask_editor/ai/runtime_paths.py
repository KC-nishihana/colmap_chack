"""
SAM Worker の一時ファイル (NPZ) 置き場の解決と古いファイルの掃除。

GUI と Worker の双方が同じ場所を指すよう、純粋な stdlib のみで実装する
(PySide6 / torch に依存しない)。

優先順位:
  1. 環境変数 COLMAP_MASK_EDITOR_RUNTIME_DIR (テスト・上書き用)
  2. %LOCALAPPDATA%/COLMAPMaskEditor/sam_runtime  (Windows ネイティブ)
  3. tempfile.gettempdir()/COLMAPMaskEditor/sam_runtime  (フォールバック)
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

ENV_OVERRIDE = "COLMAP_MASK_EDITOR_RUNTIME_DIR"
_APP_DIR = "COLMAPMaskEditor"
_RUNTIME_SUBDIR = "sam_runtime"


def get_runtime_dir(create: bool = True) -> Path:
    override = os.environ.get(ENV_OVERRIDE)
    if override:
        base = Path(override)
    else:
        local = os.environ.get("LOCALAPPDATA")
        if local:
            base = Path(local) / _APP_DIR / _RUNTIME_SUBDIR
        else:
            base = Path(tempfile.gettempdir()) / _APP_DIR / _RUNTIME_SUBDIR
    if create:
        base.mkdir(parents=True, exist_ok=True)
    return base


def cleanup_old_files(max_age_sec: float = 24 * 3600, runtime_dir: Path | None = None) -> int:
    """
    一定時間より古い一時ファイルを削除する。削除件数を返す。
    起動時に呼び、前回の取り残しを掃除する。
    """
    d = runtime_dir if runtime_dir is not None else get_runtime_dir(create=False)
    if not d.exists():
        return 0
    now = time.time()
    removed = 0
    for p in d.iterdir():
        if not p.is_file():
            continue
        try:
            if now - p.stat().st_mtime > max_age_sec:
                p.unlink()
                removed += 1
        except OSError:
            pass
    return removed


def delete_result_file(path) -> bool:
    """読み込み済みの結果ファイルを削除する。成功/不要なら True。"""
    try:
        p = Path(path)
        if p.exists():
            p.unlink()
        return True
    except OSError:
        return False
