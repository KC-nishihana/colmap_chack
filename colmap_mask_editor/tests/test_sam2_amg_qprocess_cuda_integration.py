"""
V0.8: 実 QProcess Worker 経由の RTX 4090 実機 AMG (Automatic Mask Generator) 統合テスト。

本番経路 (sam_backend/worker_main.py を実 QProcess で起動し JSON Lines + 圧縮NPZ受け渡し)
で、全画像自動分割が動作することを検証する。torch/sam2/CUDA 拡張が必須。

実行:
    $env:RUN_SAM2_CUDA_TESTS = "1"
    $env:SAM2_CHECKPOINT = "C:\\...\\sam2.1_hiera_small.pt"
    python -m pytest -m sam2_cuda colmap_mask_editor/tests/test_sam2_amg_qprocess_cuda_integration.py -v

検証 (CLAUDE.md V0.8 実機チェックリスト):
  - hello で cuda_available / cuda_extension_loaded == True / モデルロード
  - 画像ごとの generate 成功・uncompressed RLE 取得・圧縮NPZ保存・allow_pickle=False 読込
  - RLE round-trip / segment_count>0 / bbox範囲 / area一致 / predicted_iou・stability範囲
  - 3 画像独立処理
  - 途中キャンセル後も完成済み結果保持
  - 処理済み画像スキップ / 失敗画像 (retry) だけ再処理
  - Worker 終了 / GPU プロセス解放 / Worker 再起動後に単一画像推論可能
"""

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.sam2_cuda

if os.environ.get("RUN_SAM2_CUDA_TESTS") != "1":
    pytest.skip(
        "実機CUDAテストは RUN_SAM2_CUDA_TESTS=1 のときのみ実行します",
        allow_module_level=True,
    )

import cv2  # noqa: E402

from ai import amg_manifest, amg_npz, amg_rle, model_registry, protocol  # noqa: E402
from ai.amg_protocol import AmgCommand, AmgEvent  # noqa: E402
from ai.process_manager import SamProcessManager  # noqa: E402

PKG_ROOT = Path(__file__).resolve().parent.parent


def _checkpoint() -> Path:
    p = os.environ.get("SAM2_CHECKPOINT")
    if p:
        return Path(p)
    return PKG_ROOT.parent / "models" / "sam2" / "sam2.1_hiera_small.pt"


def gpu_compute_pids() -> set[int] | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None
    if out.returncode != 0:
        return None
    pids: set[int] = set()
    for line in out.stdout.splitlines():
        line = line.strip()
        if line:
            try:
                pids.add(int(line.split(",")[0].strip()))
            except (ValueError, IndexError):
                continue
    return pids


class WorkerDriver:
    """request_id 応答と job ベースイベント (event リスト) の両方を収集する。"""

    def __init__(self, qtbot):
        self.qtbot = qtbot
        self.mgr = SamProcessManager()
        self.responses: dict[int, dict] = {}
        self.events: list[dict] = []
        self.mgr.ready.connect(self._store)
        self.mgr.event_received.connect(self._store)
        self.mgr.error_received.connect(self._store)

    def _store(self, msg: dict) -> None:
        self.events.append(msg)
        rid = msg.get("request_id")
        if isinstance(rid, int):
            self.responses[rid] = msg

    def start(self) -> None:
        assert self.mgr.start(), "Worker を起動できませんでした"
        self.qtbot.waitUntil(self.mgr.is_running, timeout=30_000)

    def call(self, command: str, timeout: int = 60_000, **fields) -> dict:
        rid = self.mgr.send_command(command, **fields)
        self.qtbot.waitUntil(lambda: rid in self.responses, timeout=timeout)
        return self.responses[rid]

    def send(self, command: str, **fields) -> int:
        return self.mgr.send_command(command, **fields)

    def wait_event(self, event: str, timeout: int = 300_000) -> dict:
        self.qtbot.waitUntil(lambda: any(e.get("event") == event for e in self.events),
                             timeout=timeout)
        return next(e for e in self.events if e.get("event") == event)

    def of(self, event: str) -> list[dict]:
        return [e for e in self.events if e.get("event") == event]

    def shutdown(self) -> None:
        self.mgr.request_shutdown()
        self.qtbot.waitUntil(lambda: not self.mgr.is_running(), timeout=15_000)

    def pid(self) -> int | None:
        return self.mgr.process_id()


@pytest.fixture
def worker_factory(qtbot):
    created: list[WorkerDriver] = []

    def _make() -> WorkerDriver:
        d = WorkerDriver(qtbot)
        created.append(d)
        return d

    yield _make
    for d in created:
        try:
            d.mgr.stop(graceful_wait_ms=3000)
        except Exception:
            pass


def _make_images(tmp_path) -> tuple[Path, list[dict]]:
    """日本語+全角スペースパスに 3 枚のテスト画像を生成する。"""
    proj = tmp_path / "日本語 プロジェクト"
    img_dir = proj / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        # 画像1: 矩形 + 大円 + 小円
        lambda im: (cv2.rectangle(im, (200, 150), (900, 700), (210, 210, 210), -1),
                    cv2.circle(im, (1400, 500), 220, (120, 200, 120), -1),
                    cv2.circle(im, (1700, 250), 70, (200, 120, 120), -1)),
        # 画像2: 矩形位置変更
        lambda im: (cv2.rectangle(im, (500, 300), (1200, 800), (210, 210, 210), -1),
                    cv2.circle(im, (300, 250), 180, (120, 120, 220), -1)),
        # 画像3: 複数形状と小領域
        lambda im: ([cv2.circle(im, (200 + i * 250, 200 + (i % 2) * 300),
                                40 + i * 15, (90 + i * 20, 200, 120), -1) for i in range(6)],
                    cv2.rectangle(im, (1300, 600), (1800, 1000), (200, 200, 120), -1)),
    ]
    images = []
    for i, draw in enumerate(specs):
        im = np.zeros((1080, 1920, 3), np.uint8)
        draw(im)
        p = img_dir / f"画像 {i:03d}.png"
        ok, buf = cv2.imencode(".png", im)
        assert ok
        buf.tofile(str(p))
        images.append({"image_key": f"画像 {i:03d}.png", "source_path": str(p)})
    return proj, images


def _load_model(driver: WorkerDriver) -> dict:
    ckpt = _checkpoint()
    assert ckpt.exists(), f"チェックポイントがありません: {ckpt}"
    info = model_registry.get_model("sam2.1_hiera_small")
    res = driver.call(protocol.Command.LOAD_MODEL, timeout=180_000,
                      model_id=info.model_id, checkpoint_path=str(ckpt),
                      precision="bf16", device="cuda:0")
    assert res.get("event") == protocol.Event.MODEL_LOADED, res
    return res


def _start_amg(driver: WorkerDriver, project_root: Path, images, **extra) -> dict:
    rid = driver.send(AmgCommand.BATCH_START, project_root=str(project_root),
                      images=images, settings=amg_manifest.preset_settings("fast"),
                      preset="fast",
                      model={"model_id": "sam2.1_hiera_small",
                             "sam2_commit": "", "checkpoint_fingerprint": ""},
                      **extra)
    driver.qtbot.waitUntil(lambda: rid in driver.responses, timeout=30_000)
    started = driver.responses[rid]
    assert started.get("event") == AmgEvent.BATCH_STARTED, started
    return started


def _verify_npz(cache_dir: Path, w: int, h: int) -> int:
    data = amg_npz.verify_segments_npz(cache_dir / "segments.npz")
    n = int(data["segment_ids"].shape[0])
    assert n > 0
    # uncompressed RLE round-trip + 各種範囲
    for i in range(min(n, 5)):
        counts = amg_rle.unpack_counts(data, i)
        amg_rle.validate_rle(counts, h, w)
        assert amg_rle.rle_area(counts) == int(data["area"][i])
        mask = amg_rle.decode_rle(counts, h, w)
        assert mask.shape == (h, w) and set(np.unique(mask)).issubset({0, 255})
        bx, by, bw, bh = (int(v) for v in data["bbox_xywh"][i])
        assert 0 <= bx and 0 <= by and bx + bw <= w and by + bh <= h
        assert 0.0 <= float(data["predicted_iou"][i]) <= 1.0
        assert 0.0 <= float(data["stability_score"][i]) <= 1.0
    # NPZ に dense マスクが無い
    with np.load(cache_dir / "segments.npz", allow_pickle=False) as d:
        for name in d.files:
            assert d[name].ndim < 3
    return n


def test_amg_full_pipeline_japanese_path(worker_factory, tmp_path):
    d = worker_factory()
    d.start()
    _check = d.call(protocol.Command.HELLO, timeout=30_000)
    assert _check.get("cuda_available") is True
    assert _check.get("cuda_extension_loaded") is True
    _load_model(d)

    proj, images = _make_images(tmp_path)
    _start_amg(d, proj, images)
    d.wait_event(AmgEvent.BATCH_COMPLETED, timeout=600_000)

    assert len(d.of(AmgEvent.IMAGE_COMPLETED)) == 3  # 3 画像独立処理
    for item in images:
        cdir = amg_manifest.cache_dir_for(proj, item["image_key"])
        n = _verify_npz(cdir, 1920, 1080)
        man = amg_manifest.read_json(cdir / "manifest.json")
        assert man["segment_count"] == n and man["status"] == "ready"

    d.shutdown()


def test_amg_skip_and_retry(worker_factory, tmp_path):
    d = worker_factory()
    d.start()
    _load_model(d)
    proj, images = _make_images(tmp_path)
    _start_amg(d, proj, images)
    d.wait_event(AmgEvent.BATCH_COMPLETED, timeout=600_000)
    d.events.clear()

    # 再実行 -> 全 skip
    _start_amg(d, proj, images)
    d.wait_event(AmgEvent.BATCH_COMPLETED, timeout=120_000)
    assert len(d.of(AmgEvent.IMAGE_SKIPPED)) == 3
    assert len(d.of(AmgEvent.IMAGE_COMPLETED)) == 0
    d.events.clear()

    # retry_image (force) で 1 枚だけ再解析
    rid = d.send(AmgCommand.RETRY_IMAGE, project_root=str(proj), images=[images[0]],
                 settings=amg_manifest.preset_settings("fast"), preset="fast",
                 model={"model_id": "sam2.1_hiera_small", "sam2_commit": "", "checkpoint_fingerprint": ""})
    d.qtbot.waitUntil(lambda: rid in d.responses, timeout=30_000)
    d.wait_event(AmgEvent.BATCH_COMPLETED, timeout=300_000)
    assert len(d.of(AmgEvent.IMAGE_COMPLETED)) == 1
    d.shutdown()


def test_amg_cancel_keeps_completed_then_release(worker_factory, tmp_path):
    d = worker_factory()
    d.start()
    _load_model(d)
    proj, base = _make_images(tmp_path)
    # 枚数を増やしてキャンセル猶予を作る
    images = base * 3
    images = [{"image_key": f"{i}_{it['image_key']}", "source_path": it["source_path"]}
              for i, it in enumerate(images)]
    _start_amg(d, proj, images)
    # 最初の 1 枚完了後にキャンセル
    d.wait_event(AmgEvent.IMAGE_COMPLETED, timeout=600_000)
    d.send(AmgCommand.BATCH_CANCEL)
    d.wait_event(AmgEvent.BATCH_CANCELLED, timeout=300_000)
    completed = d.of(AmgEvent.IMAGE_COMPLETED)
    assert 0 < len(completed) < len(images)
    for ev in completed:
        cdir = amg_manifest.cache_dir_for(proj, ev["image_key"])
        amg_npz.verify_segments_npz(cdir / "segments.npz")  # 完成済み保持

    pid = d.pid()
    d.send(AmgCommand.RELEASE)
    d.wait_event(AmgEvent.RELEASED, timeout=60_000)

    # 解放後も単一画像 predict が可能 (モデル保持)
    res = d.call(protocol.Command.SET_IMAGE, timeout=120_000, image_path=base[0]["source_path"])
    assert res.get("event") == protocol.Event.IMAGE_READY, res

    d.shutdown()
    # Worker 終了で GPU PID が解放される
    pids = gpu_compute_pids()
    if pids is not None and pid is not None:
        assert pid not in pids


def test_worker_restart_single_image_after_amg(worker_factory, tmp_path):
    d = worker_factory()
    d.start()
    _load_model(d)
    proj, images = _make_images(tmp_path)
    _start_amg(d, proj, images)
    d.wait_event(AmgEvent.BATCH_COMPLETED, timeout=600_000)
    d.shutdown()

    # 再起動 -> 単一画像 SAM が使える
    d2 = worker_factory()
    d2.start()
    hello = d2.call(protocol.Command.HELLO, timeout=30_000)
    assert hello.get("cuda_extension_loaded") is True
    _load_model(d2)
    si = d2.call(protocol.Command.SET_IMAGE, timeout=120_000, image_path=images[0]["source_path"])
    assert si.get("event") == protocol.Event.IMAGE_READY
    pr = d2.call(protocol.Command.PREDICT, timeout=60_000,
                 image_key=si["image_key"], points=[[960, 540, 1]], multimask_output=True)
    assert pr.get("event") == protocol.Event.PREDICTION_READY
    d2.shutdown()
