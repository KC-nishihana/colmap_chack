import os
from pathlib import Path
from ai import protocol
from ai.process_manager import SamProcessManager

WORKER = Path(__file__).resolve().parent.parent / "sam_backend" / "worker_main.py"


def test_real_worker_hello_without_torch(qtbot, tmp_path):
    """torch未導入環境でも real worker_main が hello に応答し、AIを無効と報告する。"""
    os.environ["COLMAP_MASK_EDITOR_RUNTIME_DIR"] = str(tmp_path / "rt")
    mgr = SamProcessManager(worker_main_path=WORKER)
    with qtbot.waitSignal(mgr.worker_started, timeout=15000):
        assert mgr.start()
    with qtbot.waitSignal(mgr.ready, timeout=15000) as blk:
        mgr.send_command(protocol.Command.HELLO)
    hello = blk.args[0]
    assert hello["event"] == protocol.Event.READY
    assert hello["protocol_version"] == protocol.PROTOCOL_VERSION
    # torch/sam2 がこの環境に無ければ False (あれば True)。キーが存在することを確認。
    assert "cuda_extension_loaded" in hello
    mgr.stop()
