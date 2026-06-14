"""
GUIプロセスが torch / sam2 / sam2._C を import しないことを保証するテスト。

CLAUDE.md V0.6 の最重要要件:
  - GUIプロセスで torch, sam2, sam2._C を import しない
  - SAM処理はすべて QProcess 子プロセスで実行する

MainWindow を構築した後、sys.modules に torch / sam2 が載っていないことを確認する。
さらに ai / ui パッケージのソースに直接の import が無いことを静的に確認する。
"""

import sys
from pathlib import Path

import pytest


def test_gui_does_not_import_torch_or_sam2(qtbot):
    # 念のため事前に load されていないことを確認 (テスト環境に torch 自体未導入想定)
    from ui.main_window import MainWindow
    win = MainWindow()
    qtbot.addWidget(win)

    forbidden = ["torch", "sam2", "sam2._C", "torchvision"]
    loaded = [m for m in forbidden if m in sys.modules]
    assert loaded == [], f"GUIプロセスで禁止モジュールが読み込まれています: {loaded}"


def test_gui_modules_have_no_static_torch_import():
    """ai/ と ui/ のソースに torch / sam2 の直接 import 文が無い。"""
    pkg_root = Path(__file__).resolve().parent.parent
    targets = list((pkg_root / "ai").glob("*.py")) + list((pkg_root / "ui").glob("*.py"))
    offenders = []
    needles = ("import torch", "from torch", "import sam2", "from sam2")
    for f in targets:
        text = f.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if any(n in stripped for n in needles):
                offenders.append(f"{f.name}: {stripped}")
    assert offenders == [], "GUI側に torch/sam2 の import があります:\n" + "\n".join(offenders)
