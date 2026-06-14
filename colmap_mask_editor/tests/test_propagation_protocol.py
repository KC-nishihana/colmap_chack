"""伝播プロトコル拡張のテスト (torch不要)。"""

from ai import protocol
from ai.propagation_protocol import (
    PropagationCommand,
    PropagationDirection,
    PropagationErrorCode,
    PropagationEvent,
    make_job_error,
    make_job_event,
)


def test_command_values():
    assert PropagationCommand.START == "propagation_start"
    assert "propagation_cancel" in PropagationCommand.ALL
    assert len(PropagationCommand.ALL) == 7


def test_event_and_error_sets():
    assert PropagationEvent.FRAME_READY == "propagation_frame_ready"
    assert len(PropagationEvent.ALL) == 12
    assert len(PropagationErrorCode.ALL) == 11
    assert PropagationErrorCode.GPU_OOM == "PROPAGATION_GPU_OOM"


def test_directions():
    assert PropagationDirection.ALL == {"forward", "backward", "both"}


def test_make_job_event_has_job_id_and_protocol():
    msg = make_job_event(PropagationEvent.FRAME_READY, "prop-abc",
                         frame_index=5, foreground_ratio=0.1)
    assert msg["job_id"] == "prop-abc"
    assert msg["event"] == "propagation_frame_ready"
    assert msg["frame_index"] == 5
    assert msg["status"] == protocol.Status.OK
    assert msg["protocol_version"] == protocol.PROTOCOL_VERSION
    # 進捗イベントは request_id を持たない
    assert "request_id" not in msg


def test_make_job_event_with_request_id():
    msg = make_job_event(PropagationEvent.STARTED, "prop-x", request_id=101, frame_count=21)
    assert msg["request_id"] == 101
    assert msg["job_id"] == "prop-x"


def test_make_job_error():
    msg = make_job_error(PropagationErrorCode.BUSY, "busy", job_id="prop-1", request_id=9)
    assert protocol.is_error(msg)
    assert msg["error_code"] == "PROPAGATION_BUSY"
    assert msg["job_id"] == "prop-1"
    assert msg["request_id"] == 9


def test_make_job_error_without_job_id():
    msg = make_job_error(PropagationErrorCode.NOT_FOUND, "nope")
    assert "job_id" not in msg
