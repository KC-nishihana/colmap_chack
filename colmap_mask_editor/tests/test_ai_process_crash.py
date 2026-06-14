"""
Worker クラッシュ / タイムアウト / NPZ破損 / 古いrequest_id の障害分離テスト。

GUI (AiSession) はクラッシュしても通常マスクへ影響を与えず、ERROR へ遷移し
Worker 再起動が可能であることを確認する。
"""

import os
from pathlib import Path

import pytest

from ai import protocol
from ai.ai_session import AiSession, AiUiState

FAKE_WORKER = Path(__file__).resolve().parent / "fake_sam_worker.py"


def _make_session(tmp_path, mode="normal"):
    os.environ["FAKE_SAM_MODE"] = mode
    os.environ["COLMAP_MASK_EDITOR_RUNTIME_DIR"] = str(tmp_path / "rt")
    return AiSession(worker_main_path=FAKE_WORKER)


@pytest.fixture(autouse=True)
def _cleanup_env():
    yield
    os.environ.pop("FAKE_SAM_MODE", None)


def _wait_state(qtbot, sess, target, timeout=15000):
    if sess.state == target:
        return
    qtbot.waitUntil(lambda: sess.state == target, timeout=timeout)


def _to_prompt_editing(qtbot, sess):
    sess.start_worker()
    _wait_state(qtbot, sess, AiUiState.WORKER_READY)
    sess.load_model("sam2.1_hiera_small", "x.pt")
    _wait_state(qtbot, sess, AiUiState.MODEL_READY)
    sess.set_image("dummy.jpg")
    _wait_state(qtbot, sess, AiUiState.PROMPT_EDITING)


def test_worker_crash_goes_to_error(qtbot, tmp_path):
    sess = _make_session(tmp_path, mode="crash")
    _to_prompt_editing(qtbot, sess)
    sess.prompts.add_point(10, 10, positive=True)

    unavailable = []
    sess.worker_unavailable.connect(lambda m: unavailable.append(m))
    sess.predict()  # crash モードでは worker が os._exit(1)
    qtbot.waitUntil(lambda: sess.state == AiUiState.ERROR, timeout=15000)
    assert len(unavailable) == 1
    assert not sess.is_worker_running()


def test_worker_restart_after_crash(qtbot, tmp_path):
    sess = _make_session(tmp_path, mode="crash")
    _to_prompt_editing(qtbot, sess)
    sess.prompts.add_point(10, 10, positive=True)
    sess.predict()
    qtbot.waitUntil(lambda: sess.state == AiUiState.ERROR, timeout=15000)

    # crash モードだと再起動後も predict で落ちるので、再起動して WORKER_READY まで来ることだけ確認
    sess.restart_worker()
    _wait_state(qtbot, sess, AiUiState.WORKER_READY)
    assert sess.is_worker_running()
    sess.shutdown()


def test_predict_timeout(qtbot, tmp_path):
    sess = _make_session(tmp_path, mode="timeout")
    sess.set_timeout(protocol.Command.PREDICT, 800)  # 短いタイムアウト
    _to_prompt_editing(qtbot, sess)
    sess.prompts.add_point(10, 10, positive=True)

    errors = []
    sess.error.connect(lambda c, m: errors.append((c, m)))
    sess.predict()
    qtbot.waitUntil(lambda: len(errors) > 0, timeout=15000)
    # タイムアウト後は PROMPT_EDITING へ戻る (Worker は生存)
    assert sess.state == AiUiState.PROMPT_EDITING
    sess.shutdown()


def test_corrupt_npz_handled(qtbot, tmp_path):
    sess = _make_session(tmp_path, mode="corrupt_npz")
    _to_prompt_editing(qtbot, sess)
    sess.prompts.add_point(10, 10, positive=True)

    errors = []
    sess.error.connect(lambda c, m: errors.append((c, m)))
    sess.predict()
    qtbot.waitUntil(lambda: len(errors) > 0, timeout=15000)
    assert sess.result is None
    assert sess.state == AiUiState.PROMPT_EDITING
    sess.shutdown()


def test_missing_result_handled(qtbot, tmp_path):
    sess = _make_session(tmp_path, mode="missing_result")
    _to_prompt_editing(qtbot, sess)
    sess.prompts.add_point(10, 10, positive=True)

    errors = []
    sess.error.connect(lambda c, m: errors.append((c, m)))
    sess.predict()
    qtbot.waitUntil(lambda: len(errors) > 0, timeout=15000)
    assert sess.result is None
    sess.shutdown()


def test_stale_request_id_discarded(qtbot, tmp_path):
    """画像切替後に届いた古い予測結果は適用されない。"""
    sess = _make_session(tmp_path, mode="normal")
    _to_prompt_editing(qtbot, sess)
    sess.prompts.add_point(10, 10, positive=True)

    got = []
    sess.prediction_ready.connect(lambda r: got.append(r))

    # active_predict_id を意図的に進める (=古い結果扱いにする) ため、
    # predict 送信直後に内部 active id を別値へ。
    sess.predict()
    sess._active_predict_id = 999999  # 直後に古い扱いへ
    qtbot.wait(500)
    # prediction_ready は発火しない (古い結果として破棄)
    assert got == []
    assert sess.state in (AiUiState.PREDICTING, AiUiState.PROMPT_EDITING)
    sess.shutdown()
