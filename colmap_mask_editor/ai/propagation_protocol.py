"""
V0.7: SAM 2.1 Video Predictor によるマスク伝播の JSON Lines プロトコル拡張。

既存 ai/protocol.py を再利用し、伝播専用の Command / Event / ErrorCode を追加する。
torch / sam2 / PySide6 に依存しない純粋ロジック (GUI・Worker・テスト共有)。

設計上の要点:
  - request_id は単一コマンドの応答対応に使う (短命)。
  - job_id は伝播の開始〜完了/キャンセル/破棄まで維持する長命の識別子。
  - 長時間の進捗イベント (frame_ready/progress) は request_id を持たず job_id で追う。
"""

from __future__ import annotations

from typing import Any, Optional

from ai import protocol
from ai.protocol import PROTOCOL_VERSION, Status  # re-export 用

__all__ = [
    "PROTOCOL_VERSION",
    "PropagationCommand",
    "PropagationEvent",
    "PropagationErrorCode",
    "PropagationDirection",
    "make_job_event",
    "make_job_error",
]


class PropagationCommand:
    PREPARE = "propagation_prepare"
    START = "propagation_start"
    PAUSE = "propagation_pause"
    RESUME = "propagation_resume"
    CANCEL = "propagation_cancel"
    STATUS = "propagation_status"
    RELEASE = "propagation_release"

    ALL = frozenset({PREPARE, START, PAUSE, RESUME, CANCEL, STATUS, RELEASE})


class PropagationEvent:
    PREPARING = "propagation_preparing"
    READY = "propagation_ready"
    STARTED = "propagation_started"
    FRAME_READY = "propagation_frame_ready"
    PROGRESS = "propagation_progress"
    PAUSED = "propagation_paused"
    RESUMED = "propagation_resumed"
    CANCELLING = "propagation_cancelling"
    CANCELLED = "propagation_cancelled"
    COMPLETED = "propagation_completed"
    FAILED = "propagation_failed"
    RELEASED = "propagation_released"

    ALL = frozenset({
        PREPARING, READY, STARTED, FRAME_READY, PROGRESS, PAUSED, RESUMED,
        CANCELLING, CANCELLED, COMPLETED, FAILED, RELEASED,
    })


class PropagationErrorCode:
    BUSY = "PROPAGATION_BUSY"
    NOT_FOUND = "PROPAGATION_NOT_FOUND"
    INVALID_SEQUENCE = "PROPAGATION_INVALID_SEQUENCE"
    SIZE_MISMATCH = "PROPAGATION_SIZE_MISMATCH"
    INVALID_REFERENCE_MASK = "PROPAGATION_INVALID_REFERENCE_MASK"
    STAGE_FAILED = "PROPAGATION_STAGE_FAILED"
    INIT_FAILED = "PROPAGATION_INIT_FAILED"
    PREDICT_FAILED = "PROPAGATION_PREDICT_FAILED"
    CANCELLED = "PROPAGATION_CANCELLED"
    RESULT_WRITE_FAILED = "PROPAGATION_RESULT_WRITE_FAILED"
    GPU_OOM = "PROPAGATION_GPU_OOM"

    ALL = frozenset({
        BUSY, NOT_FOUND, INVALID_SEQUENCE, SIZE_MISMATCH, INVALID_REFERENCE_MASK,
        STAGE_FAILED, INIT_FAILED, PREDICT_FAILED, CANCELLED, RESULT_WRITE_FAILED,
        GPU_OOM,
    })


class PropagationDirection:
    FORWARD = "forward"
    BACKWARD = "backward"
    BOTH = "both"

    ALL = frozenset({FORWARD, BACKWARD, BOTH})


def make_job_event(
    event: str,
    job_id: str,
    request_id: Optional[int] = None,
    **fields: Any,
) -> dict[str, Any]:
    """job_id 付きの成功イベントを作る (進捗系は request_id 省略)。"""
    msg = protocol.make_event(event, request_id, **fields)
    msg["job_id"] = job_id
    return msg


def make_job_error(
    error_code: str,
    message: str,
    job_id: Optional[str] = None,
    request_id: Optional[int] = None,
    **fields: Any,
) -> dict[str, Any]:
    """job_id 付きのエラーイベントを作る (job_id は無いケースもある)。"""
    msg = protocol.make_error(error_code, message, request_id, **fields)
    if job_id is not None:
        msg["job_id"] = job_id
    return msg
