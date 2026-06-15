"""
V0.8: SAM 2.1 Automatic Mask Generator (AMG) 全画像自動分割の JSON Lines
プロトコル拡張。

既存 ai/protocol.py を再利用し、AMG 専用の Command / Event / ErrorCode を追加する。
torch / sam2 / PySide6 に依存しない純粋ロジック (GUI・Worker・テスト共有)。

設計上の要点 (propagation_protocol.py と同方針):
  - request_id はコマンド受付と即時応答に使う (短命)。
  - job_id は全画像バッチの開始〜完了/キャンセルまで維持する長命の識別子。
  - 画像単位の進捗イベント (image_completed / batch_progress) は request_id を
    持たず job_id で追う。古い job_id のイベントは UI へ反映しない。
"""

from __future__ import annotations

from typing import Any, Optional

from ai import protocol
from ai.protocol import PROTOCOL_VERSION, Status  # re-export 用

__all__ = [
    "PROTOCOL_VERSION",
    "AmgCommand",
    "AmgEvent",
    "AmgErrorCode",
    "AmgImageStatus",
    "make_job_event",
    "make_job_error",
]


class AmgCommand:
    """GUI -> Worker の AMG コマンド名。"""

    BATCH_START = "amg_batch_start"
    BATCH_PAUSE = "amg_batch_pause"
    BATCH_RESUME = "amg_batch_resume"
    BATCH_CANCEL = "amg_batch_cancel"
    BATCH_STATUS = "amg_batch_status"
    RETRY_IMAGE = "amg_retry_image"
    RELEASE = "amg_release"

    ALL = frozenset({
        BATCH_START, BATCH_PAUSE, BATCH_RESUME, BATCH_CANCEL,
        BATCH_STATUS, RETRY_IMAGE, RELEASE,
    })


class AmgEvent:
    """Worker -> GUI の AMG イベント名。"""

    BATCH_STARTED = "amg_batch_started"
    IMAGE_STARTED = "amg_image_started"
    IMAGE_COMPLETED = "amg_image_completed"
    IMAGE_SKIPPED = "amg_image_skipped"
    IMAGE_FAILED = "amg_image_failed"
    BATCH_PROGRESS = "amg_batch_progress"
    BATCH_PAUSED = "amg_batch_paused"
    BATCH_RESUMED = "amg_batch_resumed"
    BATCH_CANCELLING = "amg_batch_cancelling"
    BATCH_CANCELLED = "amg_batch_cancelled"
    BATCH_COMPLETED = "amg_batch_completed"
    BATCH_FAILED = "amg_batch_failed"
    RELEASED = "amg_released"

    ALL = frozenset({
        BATCH_STARTED, IMAGE_STARTED, IMAGE_COMPLETED, IMAGE_SKIPPED,
        IMAGE_FAILED, BATCH_PROGRESS, BATCH_PAUSED, BATCH_RESUMED,
        BATCH_CANCELLING, BATCH_CANCELLED, BATCH_COMPLETED, BATCH_FAILED,
        RELEASED,
    })


class AmgErrorCode:
    """AMG 関連の error_code。"""

    BUSY = "AMG_BUSY"
    MODEL_NOT_LOADED = "AMG_MODEL_NOT_LOADED"
    INVALID_SETTINGS = "AMG_INVALID_SETTINGS"
    IMAGE_LOAD_FAILED = "AMG_IMAGE_LOAD_FAILED"
    GENERATION_FAILED = "AMG_GENERATION_FAILED"
    RLE_INVALID = "AMG_RLE_INVALID"
    RESULT_WRITE_FAILED = "AMG_RESULT_WRITE_FAILED"
    CACHE_CORRUPT = "AMG_CACHE_CORRUPT"
    GPU_OOM = "AMG_GPU_OOM"
    JOB_NOT_FOUND = "AMG_JOB_NOT_FOUND"
    CANCELLED = "AMG_CANCELLED"

    ALL = frozenset({
        BUSY, MODEL_NOT_LOADED, INVALID_SETTINGS, IMAGE_LOAD_FAILED,
        GENERATION_FAILED, RLE_INVALID, RESULT_WRITE_FAILED, CACHE_CORRUPT,
        GPU_OOM, JOB_NOT_FOUND, CANCELLED,
    })


class AmgImageStatus:
    """batch_manifest / manifest が取りうる画像状態。"""

    UNPROCESSED = "unprocessed"
    PROCESSING = "processing"
    READY = "ready"
    STALE = "stale"
    FAILED = "failed"
    CANCELLED = "cancelled"
    CORRUPT = "corrupt"

    ALL = frozenset({
        UNPROCESSED, PROCESSING, READY, STALE, FAILED, CANCELLED, CORRUPT,
    })


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
    """job_id 付きのエラーイベントを作る (job_id が無いケースもある)。"""
    msg = protocol.make_error(error_code, message, request_id, **fields)
    if job_id is not None:
        msg["job_id"] = job_id
    return msg
