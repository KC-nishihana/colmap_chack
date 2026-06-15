"""V0.8: AMG プロトコル定数・メッセージ生成のテスト (torch 不要)。"""

from ai import amg_protocol as P
from ai import protocol


def test_command_event_error_sets():
    assert "amg_batch_start" in P.AmgCommand.ALL
    assert "amg_image_completed" in P.AmgEvent.ALL
    assert "AMG_GPU_OOM" in P.AmgErrorCode.ALL
    assert P.AmgImageStatus.READY in P.AmgImageStatus.ALL


def test_make_job_event_has_job_id():
    msg = P.make_job_event(P.AmgEvent.IMAGE_COMPLETED, "amg-a1", image_key="k", segment_count=84)
    assert msg["event"] == "amg_image_completed"
    assert msg["job_id"] == "amg-a1"
    assert msg["image_key"] == "k"
    assert msg["status"] == protocol.Status.OK
    assert msg["protocol_version"] == protocol.PROTOCOL_VERSION


def test_make_job_error():
    msg = P.make_job_error(P.AmgErrorCode.GPU_OOM, "oom", job_id="amg-a1", request_id=7)
    assert msg["error_code"] == "AMG_GPU_OOM"
    assert msg["job_id"] == "amg-a1"
    assert msg["request_id"] == 7
    assert protocol.is_error(msg)


def test_make_job_error_without_job_id():
    msg = P.make_job_error(P.AmgErrorCode.JOB_NOT_FOUND, "no", request_id=3)
    assert "job_id" not in msg


def test_roundtrip_encode_decode():
    msg = P.make_job_event(P.AmgEvent.BATCH_STARTED, "amg-x", request_id=1, total_images=500)
    line = protocol.encode_line(msg)
    dec = protocol.JsonLineDecoder()
    parsed = dec.feed(line + "\n")
    assert len(parsed) == 1 and parsed[0].ok
    assert parsed[0].obj["total_images"] == 500
