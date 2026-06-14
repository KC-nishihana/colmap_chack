"""
V0.7: 実 QProcess Worker 経由の SAM 2.1 Video Predictor 伝播 実機CUDAテスト。

    $env:RUN_SAM2_CUDA_TESTS = "1"
    $env:SAM2_CHECKPOINT = "C:\\...\\sam2.1_hiera_small.pt"
    python -m pytest -m sam2_cuda colmap_mask_editor/tests/test_sam2_video_qprocess_cuda_integration.py -v

検証:
  - 本番 QProcess + worker_main.py で前方向・後方向伝播
  - 5枚すべて結果取得・uint8 0/255・元サイズ・空でない
  - 日本語/全角スペースのステージング(=ソース)パスで動作
  - pause/cancel コマンドを Worker が受信できる
  - Worker 終了後の GPU プロセス解放
  - Worker 再起動後に単一画像推論が可能
"""

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.sam2_cuda

if os.environ.get("RUN_SAM2_CUDA_TESTS") != "1":
    pytest.skip("実機CUDAテストは RUN_SAM2_CUDA_TESTS=1 のときのみ実行します",
                allow_module_level=True)

import cv2  # noqa: E402

from ai import model_registry, protocol  # noqa: E402
from ai.propagation_protocol import PropagationCommand, PropagationEvent  # noqa: E402
from ai.process_manager import SamProcessManager  # noqa: E402

PKG_ROOT = Path(__file__).resolve().parent.parent
REF_FRAME = 2
XS = [200, 230, 260, 290, 320]   # 矩形のx座標。基準=index2 (x=260)
H, W = 480, 640


def _checkpoint() -> Path:
    p = os.environ.get("SAM2_CHECKPOINT")
    return Path(p) if p else PKG_ROOT.parent / "models" / "sam2" / "sam2.1_hiera_small.pt"


def gpu_compute_pids():
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15)
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None
    if out.returncode != 0:
        return None
    pids = set()
    for line in out.stdout.splitlines():
        line = line.strip()
        if line:
            try:
                pids.add(int(line.split(",")[0].strip()))
            except (ValueError, IndexError):
                pass
    return pids


class Driver:
    def __init__(self, qtbot):
        self.qtbot = qtbot
        self.mgr = SamProcessManager()
        self.by_rid: dict[int, dict] = {}
        self.events: list[dict] = []
        self.mgr.ready.connect(self._store)
        self.mgr.event_received.connect(self._store)
        self.mgr.error_received.connect(self._store)

    def _store(self, msg):
        self.events.append(msg)
        rid = msg.get("request_id")
        if isinstance(rid, int):
            self.by_rid[rid] = msg

    def start(self):
        assert self.mgr.start()
        self.qtbot.waitUntil(self.mgr.is_running, timeout=30_000)

    def call(self, command, timeout=60_000, **fields):
        rid = self.mgr.send_command(command, **fields)
        self.qtbot.waitUntil(lambda: rid in self.by_rid, timeout=timeout)
        return self.by_rid[rid]

    def wait_event(self, pred, timeout=120_000):
        self.qtbot.waitUntil(lambda: any(pred(e) for e in self.events), timeout=timeout)
        return next(e for e in self.events if pred(e))

    def shutdown(self):
        self.mgr.request_shutdown()
        self.qtbot.waitUntil(lambda: not self.mgr.is_running(), timeout=15_000)


@pytest.fixture
def driver_factory(qtbot):
    created = []

    def _make():
        d = Driver(qtbot)
        created.append(d)
        return d

    yield _make
    for d in created:
        try:
            d.mgr.stop(graceful_wait_ms=3000)
        except Exception:
            pass


@pytest.fixture
def sequence(tmp_path):
    """日本語+全角スペースパスへ5枚のソース画像と基準マスクPNGを作る。"""
    src = tmp_path / "日本語 連番"
    src.mkdir()
    frames = []
    for i, x in enumerate(XS):
        img = np.zeros((H, W, 3), np.uint8)
        cv2.rectangle(img, (x, 180), (x + 120, 320), (220, 220, 220), -1)
        p = src / f"画像 {i:03d}.jpg"
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        assert ok
        buf.tofile(str(p))
        frames.append({"frame_index": i, "entry_key": f"画像 {i:03d}.jpg", "source_path": str(p)})

    ref_mask = np.zeros((H, W), np.uint8)
    rx = XS[REF_FRAME]
    ref_mask[180:320, rx:rx + 120] = 255
    ref_path = tmp_path / "基準 マスク.png"
    ok, buf = cv2.imencode(".png", ref_mask)
    assert ok
    buf.tofile(str(ref_path))
    return frames, str(ref_path)


def _start_kwargs(frames, ref_path, direction="both"):
    return dict(
        frames=frames, reference_frame_index=REF_FRAME, reference_mask_path=ref_path,
        model_id="sam2.1_hiera_small", checkpoint_path=str(_checkpoint()),
        precision="bf16", device="cuda:0", direction=direction, max_frames=100,
    )


def test_video_propagation_both_directions(driver_factory, sequence):
    frames, ref_path = sequence
    d = driver_factory()
    d.start()
    hello = d.call(protocol.Command.HELLO, timeout=30_000)
    assert hello.get("cuda_extension_loaded") is True

    started = d.call(PropagationCommand.START, timeout=60_000, **_start_kwargs(frames, ref_path))
    assert started.get("event") == PropagationEvent.STARTED
    job_id = started["job_id"]
    assert started["frame_count"] == 5

    completed = d.wait_event(
        lambda e: e.get("event") == PropagationEvent.COMPLETED and e.get("job_id") == job_id,
        timeout=180_000,
    )
    assert completed["completed_count"] == 5

    fr = [e for e in d.events
          if e.get("event") == PropagationEvent.FRAME_READY and e.get("job_id") == job_id]
    got = {e["frame_index"]: e for e in fr}
    assert set(got) == {0, 1, 2, 3, 4}, f"covered={sorted(got)}"

    for idx, e in got.items():
        m = cv2.imdecode(np.fromfile(e["result_mask_path"], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        assert m is not None and m.shape == (H, W) and m.dtype == np.uint8
        assert set(np.unique(m)).issubset({0, 255})
        assert int((m > 0).sum()) > 0, f"frame {idx} は空マスク"

    d.shutdown()
    assert not d.mgr.is_running()


def test_video_pause_cancel_received(driver_factory, sequence):
    frames, ref_path = sequence
    d = driver_factory()
    d.start()
    assert d.call(protocol.Command.HELLO, timeout=30_000).get("cuda_extension_loaded") is True

    started = d.call(PropagationCommand.START, timeout=60_000, **_start_kwargs(frames, ref_path))
    job_id = started["job_id"]

    # pause / cancel コマンドが受信され ack が返ること
    paused = d.call(PropagationCommand.PAUSE, timeout=15_000, job_id=job_id)
    assert paused.get("event") == PropagationEvent.PAUSED
    cancelling = d.call(PropagationCommand.CANCEL, timeout=15_000, job_id=job_id)
    assert cancelling.get("event") == PropagationEvent.CANCELLING

    # ジョブは終了 (cancelled もしくは completed) する
    d.wait_event(
        lambda e: e.get("job_id") == job_id
        and e.get("event") in (PropagationEvent.CANCELLED, PropagationEvent.COMPLETED),
        timeout=180_000,
    )
    assert d.mgr.is_running()  # Worker 自体は維持
    d.shutdown()
    assert not d.mgr.is_running()


def test_video_worker_pid_release_and_restart_single_image(driver_factory, sequence):
    frames, ref_path = sequence
    d = driver_factory()
    d.start()
    assert d.call(protocol.Command.HELLO, timeout=30_000).get("cuda_extension_loaded") is True
    started = d.call(PropagationCommand.START, timeout=60_000, **_start_kwargs(frames, ref_path))
    job_id = started["job_id"]
    worker_pid = d.mgr.process_id()
    assert worker_pid and worker_pid > 0

    pids = gpu_compute_pids()
    smi = pids is not None
    if smi:
        d.qtbot.waitUntil(lambda: worker_pid in (gpu_compute_pids() or set()), timeout=30_000)

    d.wait_event(lambda e: e.get("job_id") == job_id
                 and e.get("event") == PropagationEvent.COMPLETED, timeout=180_000)
    d.shutdown()
    assert not d.mgr.is_running()
    if smi:
        d.qtbot.waitUntil(lambda: worker_pid not in (gpu_compute_pids() or set()), timeout=30_000)
        assert worker_pid not in (gpu_compute_pids() or set())

    # Worker 再起動後に単一画像推論ができる
    d2 = driver_factory()
    d2.start()
    assert d2.call(protocol.Command.HELLO, timeout=30_000).get("cuda_extension_loaded") is True
    info = model_registry.get_model("sam2.1_hiera_small")
    loaded = d2.call(protocol.Command.LOAD_MODEL, timeout=180_000,
                     model_id=info.model_id, checkpoint_path=str(_checkpoint()),
                     precision="bf16", device="cuda:0")
    assert loaded.get("event") == "model_loaded"
    img = d2.call(protocol.Command.SET_IMAGE, timeout=120_000,
                  image_path=frames[0]["source_path"])
    assert img.get("event") == "image_ready"
    pred = d2.call(protocol.Command.PREDICT, timeout=60_000, image_key=img["image_key"],
                   points=[{"x": XS[0] + 60, "y": 250, "label": 1}], multimask_output=True)
    assert pred.get("event") == "prediction_ready"
    assert 1 <= pred["mask_count"] <= 3
    Path(pred["result_path"]).unlink(missing_ok=True)
    d2.shutdown()
    assert not d2.mgr.is_running()
