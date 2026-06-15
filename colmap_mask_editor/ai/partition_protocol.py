"""
V0.9: 完全被覆 partition 生成の CPU 専用 Worker と GUI の JSON Lines プロトコル。

既存 ai/protocol.py の encode/decode/JsonLineDecoder を再利用し、partition 専用の
Command / Event / ErrorCode / Stage を追加する。torch / sam2 / PySide6 非依存。

このプロトコルで動く Worker は CPU 専用で、torch・sam2・PySide6 を import しない。
"""

from __future__ import annotations

from typing import Any, Optional

from ai import protocol
from ai.protocol import PROTOCOL_VERSION, Status  # re-export

__all__ = [
    "PROTOCOL_VERSION",
    "PartitionCommand",
    "PartitionEvent",
    "PartitionErrorCode",
    "PartitionStage",
    "make_job_event",
    "make_job_error",
]


class PartitionCommand:
    """GUI -> Worker の partition コマンド名。"""

    BUILD_START = "partition_build_start"
    BUILD_CANCEL = "partition_build_cancel"
    BUILD_STATUS = "partition_build_status"
    VALIDATE = "partition_validate"
    RELEASE = "partition_release"
    SHUTDOWN = "shutdown"

    ALL = frozenset({
        BUILD_START, BUILD_CANCEL, BUILD_STATUS, VALIDATE, RELEASE, SHUTDOWN,
    })


class PartitionEvent:
    """Worker -> GUI の partition イベント名。"""

    BUILD_STARTED = "partition_build_started"
    STAGE_CHANGED = "partition_stage_changed"
    PROGRESS = "partition_progress"
    BUILD_COMPLETED = "partition_build_completed"
    BUILD_CANCELLED = "partition_build_cancelled"
    BUILD_FAILED = "partition_build_failed"
    VALIDATED = "partition_validated"
    RELEASED = "partition_released"

    ALL = frozenset({
        BUILD_STARTED, STAGE_CHANGED, PROGRESS, BUILD_COMPLETED,
        BUILD_CANCELLED, BUILD_FAILED, VALIDATED, RELEASED,
    })


class PartitionErrorCode:
    """partition 関連の error_code。"""

    BUSY = "PARTITION_BUSY"
    INVALID_SETTINGS = "PARTITION_INVALID_SETTINGS"
    IMAGE_LOAD_FAILED = "PARTITION_IMAGE_LOAD_FAILED"
    SEGMENTS_INVALID = "PARTITION_SEGMENTS_INVALID"
    BASE_PARTITION_FAILED = "PARTITION_BASE_FAILED"
    SLIC_UNAVAILABLE = "PARTITION_SLIC_UNAVAILABLE"
    COVERAGE_INVALID = "PARTITION_COVERAGE_INVALID"
    ENCODE_FAILED = "PARTITION_ENCODE_FAILED"
    SAVE_FAILED = "PARTITION_SAVE_FAILED"
    CACHE_CORRUPT = "PARTITION_CACHE_CORRUPT"
    JOB_NOT_FOUND = "PARTITION_JOB_NOT_FOUND"
    CANCELLED = "PARTITION_CANCELLED"
    INTERNAL = "PARTITION_INTERNAL"

    ALL = frozenset({
        BUSY, INVALID_SETTINGS, IMAGE_LOAD_FAILED, SEGMENTS_INVALID,
        BASE_PARTITION_FAILED, SLIC_UNAVAILABLE, COVERAGE_INVALID,
        ENCODE_FAILED, SAVE_FAILED, CACHE_CORRUPT, JOB_NOT_FOUND,
        CANCELLED, INTERNAL,
    })


class PartitionStage:
    """partition 生成の処理ステージ。"""

    LOADING = "loading"
    BASE_PARTITION = "base_partition"
    BOUNDARY_CLEANUP = "boundary_cleanup"
    SAM_GUIDANCE = "sam_guidance"
    REGION_GRAPH = "region_graph"
    HIERARCHY_MERGE = "hierarchy_merge"
    ENCODING = "encoding"
    VALIDATION = "validation"
    SAVING = "saving"
    COMPLETED = "completed"

    ALL = frozenset({
        LOADING, BASE_PARTITION, BOUNDARY_CLEANUP, SAM_GUIDANCE, REGION_GRAPH,
        HIERARCHY_MERGE, ENCODING, VALIDATION, SAVING, COMPLETED,
    })


def make_job_event(event: str, job_id: str, request_id: Optional[int] = None,
                   **fields: Any) -> dict[str, Any]:
    msg = protocol.make_event(event, request_id, **fields)
    msg["job_id"] = job_id
    return msg


def make_job_error(error_code: str, message: str, job_id: Optional[str] = None,
                   request_id: Optional[int] = None, **fields: Any) -> dict[str, Any]:
    msg = protocol.make_error(error_code, message, request_id, **fields)
    if job_id is not None:
        msg["job_id"] = job_id
    return msg
