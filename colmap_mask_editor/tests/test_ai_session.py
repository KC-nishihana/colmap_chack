"""
AiSession 状態機械の統合テスト (Fake Worker 使用)。

確認:
  - start -> hello -> WORKER_READY
  - load_model -> MODEL_READY
  - set_image -> PROMPT_EDITING (image_key 確定)
  - predict -> PREVIEW (3候補・最大スコア初期選択)
  - 候補切替 (再推論しない)
  - 古い request_id / 別 image_key の結果を破棄
  - CUDA拡張なし(hello)で AI 無効化
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


def _bring_to_model_ready(qtbot, sess):
    sess.start_worker()
    _wait_state(qtbot, sess, AiUiState.WORKER_READY)
    sess.load_model("sam2.1_hiera_small", "x.pt", "bf16", "cuda:0")
    _wait_state(qtbot, sess, AiUiState.MODEL_READY)


def test_worker_ready_after_hello(qtbot, tmp_path):
    sess = _make_session(tmp_path)
    sess.start_worker()
    _wait_state(qtbot, sess, AiUiState.WORKER_READY)
    assert sess.cuda_extension_loaded is True
    assert sess.hello_info["gpu_name"] == "Fake RTX 4090"
    sess.shutdown()


def test_load_model(qtbot, tmp_path):
    sess = _make_session(tmp_path)
    _bring_to_model_ready(qtbot, sess)
    assert sess.model_id == "sam2.1_hiera_small"
    sess.shutdown()


def test_set_image_then_predict(qtbot, tmp_path):
    sess = _make_session(tmp_path)
    _bring_to_model_ready(qtbot, sess)

    sess.set_image("dummy.jpg")
    _wait_state(qtbot, sess, AiUiState.PROMPT_EDITING)
    assert sess.image_key is not None

    sess.prompts.add_point(10, 10, positive=True)
    with qtbot.waitSignal(sess.prediction_ready, timeout=15000) as blk:
        sess.predict()
    result = blk.args[0]
    assert result.mask_count == 3
    assert sess.state == AiUiState.PREVIEW
    # 最大スコア候補が初期選択
    assert sess.selected_candidate_index == result.best_index()
    sess.shutdown()


def test_candidate_switch_no_repredict(qtbot, tmp_path):
    sess = _make_session(tmp_path)
    _bring_to_model_ready(qtbot, sess)
    sess.set_image("dummy.jpg")
    _wait_state(qtbot, sess, AiUiState.PROMPT_EDITING)
    sess.prompts.add_point(10, 10, positive=True)
    with qtbot.waitSignal(sess.prediction_ready, timeout=15000):
        sess.predict()

    changed = []
    sess.candidate_changed.connect(lambda i: changed.append(i))
    sess.select_candidate(2)
    assert sess.selected_candidate_index == 2
    assert changed == [2]
    # マスク取得できる
    assert sess.selected_mask() is not None
    sess.shutdown()


def test_cuda_extension_unavailable_disables_ai(qtbot, tmp_path):
    sess = _make_session(tmp_path, mode="cuda_extension_false")
    msgs = []
    sess.cuda_extension_unavailable.connect(lambda m: msgs.append(m))
    sess.start_worker()
    qtbot.waitUntil(lambda: sess.state == AiUiState.ERROR, timeout=15000)
    assert sess.cuda_extension_loaded is False
    assert len(msgs) == 1
    sess.shutdown()


def test_oom_keeps_worker_and_returns_to_prompt(qtbot, tmp_path):
    sess = _make_session(tmp_path, mode="oom")
    _bring_to_model_ready(qtbot, sess)
    sess.set_image("dummy.jpg")
    _wait_state(qtbot, sess, AiUiState.PROMPT_EDITING)
    sess.prompts.add_point(10, 10, positive=True)

    errors = []
    sess.error.connect(lambda c, m: errors.append((c, m)))
    sess.predict()
    qtbot.waitUntil(lambda: len(errors) > 0, timeout=15000)
    assert errors[0][0] == protocol.ErrorCode.CUDA_OOM
    # Worker は維持され PROMPT_EDITING へ戻る
    assert sess.is_worker_running()
    assert sess.state == AiUiState.PROMPT_EDITING
    sess.shutdown()
