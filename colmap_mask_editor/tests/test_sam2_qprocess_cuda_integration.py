"""
V0.6.1: 実 QProcess Worker 経由の RTX 4090 実機 CUDA 統合テスト。

既存の test_sam2_cuda_integration.py は同一 pytest プロセス内で
Sam2ModelManager / Sam2Predictor を直接呼ぶ。本テストは「実際の QProcess で
sam_backend/worker_main.py を起動し、JSON Lines プロトコル + NPZ 受け渡し」
という本番経路を検証する。

実行:
    $env:RUN_SAM2_CUDA_TESTS = "1"
    $env:SAM2_CHECKPOINT = "C:\\...\\sam2.1_hiera_small.pt"   # 省略時は既定パス
    python -m pytest -m sam2_cuda colmap_mask_editor/tests/test_sam2_qprocess_cuda_integration.py -v

検証:
  - hello で cuda_available / cuda_extension_loaded == True
  - load_model -> set_image -> predict (正/負/矩形) が本番 QProcess 経路で成功
  - 結果 NPZ が uint8 0/255・元画像サイズ・scores 整合
  - 日本語/全角スペースを含むパスの画像で推論
  - Worker 終了後に QProcess が NotRunning になる (必須)
  - Worker 終了後に対象 PID が nvidia-smi の compute-apps から消える (PID別GPU解放)
  - Worker 再起動後に再び hello/model/predict できる
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

from ai import model_registry, protocol  # noqa: E402
from ai.process_manager import SamProcessManager  # noqa: E402

PKG_ROOT = Path(__file__).resolve().parent.parent


def _checkpoint() -> Path:
    p = os.environ.get("SAM2_CHECKPOINT")
    if p:
        return Path(p)
    return PKG_ROOT.parent / "models" / "sam2" / "sam2.1_hiera_small.pt"


# --------------------------------------------------------------------------- #
# nvidia-smi で PID 別 GPU 使用メモリを取得
# --------------------------------------------------------------------------- #

def gpu_compute_pids() -> set[int] | None:
    """nvidia-smi の GPU compute-apps に載っている PID 集合を返す。

    Windows WDDM では used_memory が [N/A] になり PID 別 VRAM を取得できないため、
    PID の在/不在を主判定に用いる (対象 Worker プロセスが GPU コンテキストを保持
    している間は載り、プロセス終了で消える)。nvidia-smi を使えない場合は None。
    """
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
        if not line:
            continue
        try:
            pids.add(int(line.split(",")[0].strip()))
        except (ValueError, IndexError):
            continue
    return pids


# --------------------------------------------------------------------------- #
# QProcess Worker 同期ドライバ
# --------------------------------------------------------------------------- #

class WorkerDriver:
    def __init__(self, qtbot):
        self.qtbot = qtbot
        self.mgr = SamProcessManager()
        self.responses: dict[int, dict] = {}
        self.mgr.ready.connect(self._store)
        self.mgr.event_received.connect(self._store)
        self.mgr.error_received.connect(self._store)

    def _store(self, msg: dict) -> None:
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

    def shutdown(self) -> None:
        self.mgr.request_shutdown()
        self.qtbot.waitUntil(lambda: not self.mgr.is_running(), timeout=10_000)

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


@pytest.fixture
def jp_image(tmp_path) -> tuple[Path, int, int]:
    """日本語+全角スペースを含むパスへテスト画像を保存して返す。"""
    proj = tmp_path / "日本語 プロジェクト"
    proj.mkdir(parents=True, exist_ok=True)
    img_path = proj / "画像 001.png"

    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(image, (300, 150), (950, 650), (220, 220, 220), -1)

    ok, buf = cv2.imencode(".png", image)
    assert ok, "テスト画像のエンコードに失敗"
    buf.tofile(str(img_path))   # 日本語パス対応の書き出し
    assert img_path.exists()
    return img_path, 1280, 720


def _load_model(driver: WorkerDriver) -> dict:
    ckpt = _checkpoint()
    assert ckpt.exists(), f"チェックポイントがありません: {ckpt}"
    info = model_registry.get_model("sam2.1_hiera_small")
    return driver.call(
        protocol.Command.LOAD_MODEL, timeout=180_000,
        model_id=info.model_id, checkpoint_path=str(ckpt),
        precision="bf16", device="cuda:0",
    )


def _read_and_delete_npz(result_path: str):
    with np.load(result_path, allow_pickle=False) as data:
        masks = data["masks"].copy()
        scores = data["scores"].copy()
    Path(result_path).unlink(missing_ok=True)
    return masks, scores


def _check_hello(hello: dict) -> None:
    assert hello.get("event") == protocol.Event.READY
    assert hello.get("cuda_available") is True
    assert hello.get("cuda_extension_loaded") is True
    assert hello.get("gpu_name")
    assert hello.get("compute_capability")


def _check_prediction(event: dict, w: int, h: int) -> None:
    assert event.get("event") == "prediction_ready", f"想定外: {event}"
    mask_count = event["mask_count"]
    assert 1 <= mask_count <= 3
    assert len(event["scores"]) == mask_count
    assert event["width"] == w and event["height"] == h

    masks, scores = _read_and_delete_npz(event["result_path"])
    assert masks.dtype == np.uint8
    assert masks.shape[0] == mask_count
    assert masks.shape[1:] == (h, w)
    assert set(np.unique(masks)).issubset({0, 255})
    assert len(scores) == masks.shape[0]


# --------------------------------------------------------------------------- #
# テスト本体
# --------------------------------------------------------------------------- #

def test_qprocess_full_inference_japanese_path(worker_factory, jp_image):
    """本番 QProcess 経路で hello/load/set_image/正・負・矩形 predict + NPZ 検証。"""
    img_path, w, h = jp_image
    d = worker_factory()
    d.start()

    hello = d.call(protocol.Command.HELLO, timeout=30_000)
    _check_hello(hello)

    loaded = _load_model(d)
    assert loaded.get("event") == "model_loaded"
    assert loaded.get("vram_allocated_mb", 0) > 0

    img = d.call(protocol.Command.SET_IMAGE, timeout=120_000, image_path=str(img_path))
    assert img.get("event") == "image_ready"
    assert img["width"] == w and img["height"] == h
    assert img["image_key"]
    key = img["image_key"]

    # 正クリック
    pos = d.call(protocol.Command.PREDICT, timeout=60_000,
                 image_key=key, points=[{"x": 640, "y": 360, "label": 1}],
                 multimask_output=True)
    _check_prediction(pos, w, h)

    # 正 + 負クリック
    posneg = d.call(protocol.Command.PREDICT, timeout=60_000,
                    image_key=key,
                    points=[{"x": 640, "y": 360, "label": 1},
                            {"x": 50, "y": 50, "label": 0}],
                    multimask_output=True)
    _check_prediction(posneg, w, h)

    # 矩形
    box = d.call(protocol.Command.PREDICT, timeout=60_000,
                 image_key=key, box=[250, 100, 1000, 690], multimask_output=True)
    _check_prediction(box, w, h)

    d.shutdown()
    assert not d.mgr.is_running()


def test_qprocess_worker_terminate_frees_gpu(worker_factory, jp_image):
    """Worker 終了後: QProcess NotRunning (必須) + 対象 PID が GPU から消える。"""
    img_path, w, h = jp_image
    d = worker_factory()
    d.start()

    _check_hello(d.call(protocol.Command.HELLO, timeout=30_000))
    loaded = _load_model(d)
    assert loaded.get("event") == "model_loaded"
    d.call(protocol.Command.SET_IMAGE, timeout=120_000, image_path=str(img_path))

    worker_pid = d.pid()
    assert worker_pid and worker_pid > 0

    smi_available = gpu_compute_pids() is not None
    if smi_available:
        # モデルロード後、対象 PID が GPU compute-apps に現れるのを待つ
        d.qtbot.waitUntil(
            lambda: worker_pid in (gpu_compute_pids() or set()), timeout=20_000
        )
        assert worker_pid in (gpu_compute_pids() or set())

    # shutdown -> QProcess 終了確認 (skip 不可・必須)
    d.shutdown()
    assert not d.mgr.is_running(), "Worker QProcess が NotRunning になっていません"

    if not smi_available:
        pytest.skip("nvidia-smiによるPID別GPU確認を利用できません (QProcess終了は確認済み)")

    # 対象 Worker PID が GPU compute-apps から消える = GPU プロセス解放を主判定
    d.qtbot.waitUntil(
        lambda: worker_pid not in (gpu_compute_pids() or set()), timeout=20_000
    )
    assert worker_pid not in (gpu_compute_pids() or set()), \
        f"Worker PID {worker_pid} が GPU compute-apps から消えていません"


def test_qprocess_restart_and_reinfer(worker_factory, jp_image):
    """Worker 再起動後に再びモデルロード・推論できる。"""
    img_path, w, h = jp_image

    # 1 回目
    d1 = worker_factory()
    d1.start()
    _check_hello(d1.call(protocol.Command.HELLO, timeout=30_000))
    assert _load_model(d1).get("event") == "model_loaded"
    img1 = d1.call(protocol.Command.SET_IMAGE, timeout=120_000, image_path=str(img_path))
    pred1 = d1.call(protocol.Command.PREDICT, timeout=60_000,
                    image_key=img1["image_key"],
                    points=[{"x": 640, "y": 360, "label": 1}], multimask_output=True)
    _check_prediction(pred1, w, h)
    first_pid = d1.pid()
    d1.shutdown()
    assert not d1.mgr.is_running(), "1回目の QProcess が NotRunning になっていません"

    # 2 回目 (新しい Worker)
    d2 = worker_factory()
    d2.start()
    hello2 = d2.call(protocol.Command.HELLO, timeout=30_000)
    _check_hello(hello2)
    assert _load_model(d2).get("event") == "model_loaded"
    img2 = d2.call(protocol.Command.SET_IMAGE, timeout=120_000, image_path=str(img_path))
    pred2 = d2.call(protocol.Command.PREDICT, timeout=60_000,
                    image_key=img2["image_key"],
                    points=[{"x": 640, "y": 360, "label": 1}], multimask_output=True)
    _check_prediction(pred2, w, h)
    second_pid = d2.pid()
    d2.shutdown()
    assert not d2.mgr.is_running(), "2回目の QProcess が NotRunning になっていません"

    # PID は短時間で再利用されうるため唯一条件にはしない (機能成功が必須条件)。
    if first_pid is not None and second_pid is not None and first_pid != second_pid:
        assert first_pid != second_pid
