"""
Worker のログ設定。

重要: stdout は JSON Lines 専用なので、ログは必ず stderr とログファイルへ出す。
(CLAUDE.md V0.6 要件: ログ・警告・トレースバックは標準エラーまたはログファイルへ)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


def setup_worker_logging(log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    root = logging.getLogger()
    root.setLevel(level)

    # 既存ハンドラを除去 (二重出力防止)
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # stderr ハンドラ
    stderr_handler = logging.StreamHandler(stream=sys.stderr)
    stderr_handler.setFormatter(fmt)
    root.addHandler(stderr_handler)

    # ファイルハンドラ (任意)
    if log_file is None:
        local = os.environ.get("LOCALAPPDATA")
        if local:
            log_dir = Path(local) / "COLMAPMaskEditor" / "logs"
        else:
            log_dir = Path.cwd() / "logs"
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / "sam_worker.log"
        except OSError:
            log_file = None

    if log_file is not None:
        try:
            file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
            file_handler.setFormatter(fmt)
            root.addHandler(file_handler)
        except OSError:
            pass

    return logging.getLogger("sam_worker")
