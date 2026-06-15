"""V0.9: CPU 専用 partition Worker を実 subprocess で起動した統合テスト。

torch / sam2 / PySide6 を import しないこと、JSON Lines で build が完走し
partition.npz が生成されることを確認する。
"""

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from ai import protocol, partition_npz
from ai.partition_protocol import PartitionCommand, PartitionEvent
from ai import partition_manifest as pman

from tests._partition_helpers import synthetic_bgr

PKG_ROOT = Path(__file__).resolve().parent.parent  # colmap_mask_editor/


def _env():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PKG_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def test_worker_does_not_import_torch_sam2_pyside6():
    code = (
        "import sys;"
        "import partition_backend.partition_worker_main as m;"
        "bad=[x for x in ('torch','sam2','sam2._C','PySide6') if x in sys.modules];"
        "print('BAD' if bad else 'OK', bad)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=_env(), capture_output=True, text=True, timeout=120,
    )
    assert "OK" in proc.stdout, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"


class _Worker:
    """Worker subprocess を起動し、stdout を専用スレッドで Queue へ読み出す。

    pytest ランナー下でメインスレッドの blocking readline が返らない問題を避けるため、
    読み出しは daemon スレッドへ分離する (Worker 自体は driver で完走を確認済み)。
    stderr は一時ファイルへ退避し、失敗時の診断に使う。
    """

    def __init__(self, tmp_path):
        self._err_path = tmp_path / "worker_stderr.log"
        self._err = open(self._err_path, "wb")
        # Worker は PYTHONIOENCODING=utf-8 で stdin/stdout を扱うため、パイプも
        # utf-8 に固定する (text=True 既定の locale=cp932 だと日本語パスを含む
        # コマンドで worker 側が utf-8 デコードに失敗する)。
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "partition_backend.partition_worker_main"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._err,
            env=_env(), text=True, encoding="utf-8", bufsize=1,
        )
        self._q: "queue.Queue[dict | None]" = queue.Queue()
        self._t = threading.Thread(target=self._reader, daemon=True)
        self._t.start()

    def _reader(self):
        for line in iter(self.proc.stdout.readline, ""):
            line = line.strip()
            if not line:
                continue
            try:
                self._q.put(json.loads(line))
            except ValueError:
                continue
        self._q.put(None)

    def send(self, msg):
        self.proc.stdin.write(protocol.encode_line(msg) + "\n")
        self.proc.stdin.flush()

    def read_until(self, events, timeout=60):
        deadline = time.time() + timeout
        collected = []
        while time.time() < deadline:
            try:
                obj = self._q.get(timeout=max(0.01, deadline - time.time()))
            except queue.Empty:
                break
            if obj is None:
                break
            collected.append(obj)
            if obj.get("event") in events:
                return obj, collected
        try:
            self._err.flush()
            err = self._err_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        except Exception:
            err = "(stderr 読めず)"
        raise AssertionError(
            f"イベント {events} を受信できませんでした: {collected}\n"
            f"worker stderr:\n{err}")

    def close(self):
        try:
            self.send(protocol.make_request(PartitionCommand.SHUTDOWN, 99))
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()
        finally:
            try:
                self._err.close()
            except Exception:
                pass


def test_worker_build_end_to_end(tmp_path):
    img = synthetic_bgr(120, 160, seed=8)
    img_path = tmp_path / "テスト 画像.png"
    ok, buf = cv2.imencode(".png", img)
    img_path.write_bytes(buf.tobytes())
    out_dir = tmp_path / "cache"

    w = _Worker(tmp_path)
    try:
        w.read_until({PartitionEvent.RELEASED})  # ready
        w.send(protocol.make_request(PartitionCommand.BUILD_START, 1,
               job_id="job1", image_path=str(img_path), image_key="k",
               output_dir=str(out_dir),
               settings={"backend": "auto", "working_max_side": 0,
                         "base_region_count": 50, "default_visible_count": 20,
                         "min_region_area_ratio": 10}))
        done, events = w.read_until(
            {PartitionEvent.BUILD_COMPLETED, PartitionEvent.BUILD_FAILED})
        assert done["event"] == PartitionEvent.BUILD_COMPLETED, done
        assert done["coverage_ratio"] == 1.0
        stages = [e.get("stage") for e in events if e.get("event") == PartitionEvent.STAGE_CHANGED]
        assert "completed" in stages
        partition_npz.verify_partition_npz(out_dir / pman.PARTITION_NPZ_NAME)
    finally:
        w.close()


def test_worker_validate(tmp_path):
    from tests._partition_helpers import simple_three_leaf
    p = tmp_path / "partition.npz"
    partition_npz.save_partition_npz(p, simple_three_leaf())
    w = _Worker(tmp_path)
    try:
        w.read_until({PartitionEvent.RELEASED})
        w.send(protocol.make_request(PartitionCommand.VALIDATE, 2,
               partition_path=str(p)))
        ev, _ = w.read_until({PartitionEvent.VALIDATED})
        assert ev["valid"] is True
    finally:
        w.close()
