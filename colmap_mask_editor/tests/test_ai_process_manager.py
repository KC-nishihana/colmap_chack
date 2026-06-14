"""
SamProcessManager の QProcess 統合テスト (Fake Worker 使用・torch不要)。

確認:
  - Worker 起動 / hello 往復
  - request_id 採番
  - stdout (JSON) と stderr (ログ) の分離
  - 不正JSON行の破棄 (落ちない)
  - predict 結果イベント
  - Worker 正常終了
"""

import os
from pathlib import Path

import pytest

from ai import protocol
from ai.process_manager import SamProcessManager

FAKE_WORKER = Path(__file__).resolve().parent / "fake_sam_worker.py"


def _make_manager(tmp_path, mode="normal"):
    os.environ["FAKE_SAM_MODE"] = mode
    os.environ["COLMAP_MASK_EDITOR_RUNTIME_DIR"] = str(tmp_path / "rt")
    return SamProcessManager(worker_main_path=FAKE_WORKER)


def _start(qtbot, mgr):
    with qtbot.waitSignal(mgr.worker_started, timeout=15000):
        assert mgr.start()


def _send_wait(qtbot, mgr, signal, command, **fields):
    with qtbot.waitSignal(signal, timeout=15000) as blk:
        mgr.send_command(command, **fields)
    return blk.args[0]


@pytest.fixture(autouse=True)
def _cleanup_env():
    yield
    os.environ.pop("FAKE_SAM_MODE", None)


def test_start_and_hello(qtbot, tmp_path):
    mgr = _make_manager(tmp_path)
    _start(qtbot, mgr)
    hello = _send_wait(qtbot, mgr, mgr.ready, protocol.Command.HELLO)
    assert hello["event"] == protocol.Event.READY
    assert hello["cuda_extension_loaded"] is True
    assert hello["gpu_name"] == "Fake RTX 4090"
    mgr.stop()


def test_request_id_increments(qtbot, tmp_path):
    mgr = _make_manager(tmp_path)
    assert mgr.next_request_id() == 1
    assert mgr.next_request_id() == 2
    assert mgr.next_request_id() == 3


def test_predict_flow(qtbot, tmp_path):
    mgr = _make_manager(tmp_path)
    _start(qtbot, mgr)
    _send_wait(qtbot, mgr, mgr.ready, protocol.Command.HELLO)

    loaded = _send_wait(qtbot, mgr, mgr.event_received, protocol.Command.LOAD_MODEL,
                        model_id="sam2.1_hiera_small", checkpoint_path="x.pt",
                        precision="bf16", device="cuda:0")
    assert loaded["event"] == protocol.Event.MODEL_LOADED

    img = _send_wait(qtbot, mgr, mgr.event_received, protocol.Command.SET_IMAGE,
                     image_path="dummy.jpg")
    assert img["event"] == protocol.Event.IMAGE_READY
    key = img["image_key"]

    pred = _send_wait(qtbot, mgr, mgr.event_received, protocol.Command.PREDICT,
                      image_key=key, points=[{"x": 10, "y": 10, "label": 1}],
                      multimask_output=True)
    assert pred["event"] == protocol.Event.PREDICTION_READY
    assert pred["mask_count"] == 3
    assert Path(pred["result_path"]).exists()
    mgr.stop()


def test_stderr_separated_from_stdout(qtbot, tmp_path):
    """stderr_noise モードでも stdout の JSON が壊れず hello が届く。"""
    mgr = _make_manager(tmp_path, mode="stderr_noise")
    log_lines = []
    mgr.log_line.connect(lambda s: log_lines.append(s))
    _start(qtbot, mgr)
    hello = _send_wait(qtbot, mgr, mgr.ready, protocol.Command.HELLO)
    assert hello["event"] == protocol.Event.READY
    qtbot.wait(300)
    assert any("noise" in s for s in log_lines)
    mgr.stop()


def test_invalid_json_line_discarded(qtbot, tmp_path):
    """invalid_json モード: predict 前に非JSON行が来ても prediction_ready は届く。"""
    mgr = _make_manager(tmp_path, mode="invalid_json")
    _start(qtbot, mgr)
    _send_wait(qtbot, mgr, mgr.ready, protocol.Command.HELLO)
    _send_wait(qtbot, mgr, mgr.event_received, protocol.Command.LOAD_MODEL, checkpoint_path="x.pt")
    img = _send_wait(qtbot, mgr, mgr.event_received, protocol.Command.SET_IMAGE, image_path="d.jpg")
    key = img["image_key"]
    pred = _send_wait(qtbot, mgr, mgr.event_received, protocol.Command.PREDICT,
                      image_key=key, points=[{"x": 1, "y": 1, "label": 1}])
    assert pred["event"] == protocol.Event.PREDICTION_READY
    mgr.stop()


def test_graceful_stop(qtbot, tmp_path):
    mgr = _make_manager(tmp_path)
    _start(qtbot, mgr)
    assert mgr.is_running()
    mgr.stop()
    assert not mgr.is_running()
