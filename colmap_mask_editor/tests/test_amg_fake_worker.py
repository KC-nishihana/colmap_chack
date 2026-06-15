"""
V0.8: Fake Worker を実 subprocess として起動し、AMG プロトコルを end-to-end 検証する
(torch / CUDA 不要)。実 AmgBatchRunner が動き、実 NPZ / manifest を書く。
"""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from ai import amg_manifest as M, amg_npz
from ai.amg_protocol import AmgCommand, AmgEvent
from ai.protocol import make_request

_FAKE = str(Path(__file__).parent / "fake_sam_worker.py")
_PKG = str(Path(__file__).resolve().parent.parent)


class _Worker:
    def __init__(self, mode):
        env = dict(os.environ)
        env["FAKE_SAM_MODE"] = mode
        env["PYTHONPATH"] = _PKG + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONUNBUFFERED"] = "1"
        self.p = subprocess.Popen(
            [sys.executable, "-u", _FAKE],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env=env, text=True, encoding="utf-8", bufsize=1,
        )
        self.events = []
        self._lock = threading.Lock()
        self._t = threading.Thread(target=self._reader, daemon=True)
        self._t.start()

    def _reader(self):
        for line in self.p.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            with self._lock:
                self.events.append(obj)

    def send(self, command, request_id, **fields):
        self.p.stdin.write(json.dumps(make_request(command, request_id, **fields)) + "\n")
        self.p.stdin.flush()

    def wait_for(self, event, timeout=20):
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                for e in self.events:
                    if e.get("event") == event:
                        return e
            if self.p.poll() is not None:
                # プロセス終了。残りを少し待つ
                time.sleep(0.1)
                with self._lock:
                    for e in self.events:
                        if e.get("event") == event:
                            return e
                return None
            time.sleep(0.02)
        return None

    def of(self, event):
        with self._lock:
            return [e for e in self.events if e.get("event") == event]

    def close(self):
        try:
            self.send(AmgCommand.RELEASE, 999)
            self.p.stdin.write(json.dumps(make_request("shutdown", 1000)) + "\n")
            self.p.stdin.flush()
        except Exception:
            pass
        try:
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


def _images(tmp_path, n):
    out = []
    for i in range(n):
        p = tmp_path / "src" / f"IMG_{i:03d}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\xff\xd8\xff" + bytes([i]) * 200)
        out.append({"image_key": f"IMG_{i:03d}.jpg", "source_path": str(p)})
    return out


def _start(w, tmp_path, images, rid=1, **extra):
    w.send(AmgCommand.BATCH_START, rid, project_root=str(tmp_path),
           images=images, settings=M.preset_settings("fast"), preset="fast",
           model={"model_id": "fake", "sam2_commit": "fake", "checkpoint_fingerprint": "fp"},
           **extra)


@pytest.fixture
def worker_factory():
    workers = []

    def make(mode):
        w = _Worker(mode)
        workers.append(w)
        return w

    yield make
    for w in workers:
        w.close()


def test_amg_normal_end_to_end(worker_factory, tmp_path):
    w = worker_factory("amg_normal")
    images = _images(tmp_path, 3)
    _start(w, tmp_path, images)
    started = w.wait_for(AmgEvent.BATCH_STARTED)
    assert started is not None and started["total_images"] == 3
    assert started["request_id"] == 1 and "job_id" in started
    done = w.wait_for(AmgEvent.BATCH_COMPLETED)
    assert done is not None
    assert len(w.of(AmgEvent.IMAGE_COMPLETED)) == 3
    # 実 NPZ / manifest が書かれ検証できる
    for item in images:
        cdir = M.cache_dir_for(tmp_path, item["image_key"])
        amg_npz.verify_segments_npz(cdir / "segments.npz")
        assert M.read_json(cdir / "manifest.json")["segment_count"] == 2


def test_amg_skip_on_rerun(worker_factory, tmp_path):
    images = _images(tmp_path, 2)
    w1 = worker_factory("amg_normal")
    _start(w1, tmp_path, images)
    assert w1.wait_for(AmgEvent.BATCH_COMPLETED) is not None
    # 2 回目: 別 worker でも cache 再利用で skip
    w2 = worker_factory("amg_normal")
    _start(w2, tmp_path, images, rid=2)
    assert w2.wait_for(AmgEvent.BATCH_COMPLETED) is not None
    assert len(w2.of(AmgEvent.IMAGE_SKIPPED)) == 2
    assert len(w2.of(AmgEvent.IMAGE_COMPLETED)) == 0


def test_amg_one_image_failure_isolated(worker_factory, tmp_path):
    w = worker_factory("amg_one_image_failure")
    images = _images(tmp_path, 3)
    _start(w, tmp_path, images)
    assert w.wait_for(AmgEvent.BATCH_COMPLETED) is not None
    assert len(w.of(AmgEvent.IMAGE_FAILED)) == 1
    assert len(w.of(AmgEvent.IMAGE_COMPLETED)) == 2


def test_amg_cancel_keeps_completed(worker_factory, tmp_path):
    w = worker_factory("amg_slow")
    images = _images(tmp_path, 8)
    _start(w, tmp_path, images)
    assert w.wait_for(AmgEvent.BATCH_STARTED) is not None
    # 最初の画像完了を待ってからキャンセル
    w.wait_for(AmgEvent.IMAGE_COMPLETED, timeout=10)
    w.send(AmgCommand.BATCH_CANCEL, 50)
    cancelled = w.wait_for(AmgEvent.BATCH_CANCELLED, timeout=15)
    assert cancelled is not None
    completed = w.of(AmgEvent.IMAGE_COMPLETED)
    assert len(completed) < 8
    for ev in completed:
        cdir = M.cache_dir_for(tmp_path, ev["image_key"])
        amg_npz.verify_segments_npz(cdir / "segments.npz")  # 完成済みは保持


def test_amg_pause_resume(worker_factory, tmp_path):
    w = worker_factory("amg_slow")
    images = _images(tmp_path, 6)
    _start(w, tmp_path, images)
    assert w.wait_for(AmgEvent.BATCH_STARTED) is not None
    w.wait_for(AmgEvent.IMAGE_COMPLETED, timeout=10)
    w.send(AmgCommand.BATCH_PAUSE, 60)
    paused = w.wait_for(AmgEvent.BATCH_PAUSED, timeout=10)
    assert paused is not None
    w.send(AmgCommand.BATCH_RESUME, 61)
    assert w.wait_for(AmgEvent.BATCH_RESUMED, timeout=10) is not None
    assert w.wait_for(AmgEvent.BATCH_COMPLETED, timeout=20) is not None


def test_amg_oom_retry_success(worker_factory, tmp_path):
    w = worker_factory("amg_oom_retry_success")
    images = _images(tmp_path, 1)
    _start(w, tmp_path, images)
    assert w.wait_for(AmgEvent.BATCH_COMPLETED) is not None
    cdir = M.cache_dir_for(tmp_path, images[0]["image_key"])
    man = M.read_json(cdir / "manifest.json")
    assert man["generator"]["points_per_batch"] == 16  # 縮小値を記録
