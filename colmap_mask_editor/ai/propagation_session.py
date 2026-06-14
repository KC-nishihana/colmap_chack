"""
V0.7: 伝播セッションの状態モデル (GUI 側・torch非依存)。

GUI は torch Tensor や SAM inference state を保持しない。フレームの状態・品質指標・
採否・結果PNGのパスのみを保持する。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from ai.propagation_order import PropagationOrder  # re-export 兼用


class PropagationUiState(Enum):
    IDLE = auto()
    PREFLIGHT = auto()
    STAGING = auto()
    INITIALIZING = auto()
    RUNNING = auto()
    PAUSED = auto()
    CANCELLING = auto()
    CANCELLED = auto()
    REVIEW = auto()
    APPLYING = auto()
    COMPLETED = auto()
    ERROR = auto()


# フレーム処理状態 (PropagationFrame.state の値)
class FrameState:
    PENDING = "pending"
    DONE = "done"
    WARNING = "warning"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PropagationFrame:
    frame_index: int
    entry_key: str
    source_path: str
    staged_path: Optional[str] = None
    result_mask_path: Optional[str] = None

    state: str = FrameState.PENDING
    warning_codes: list[str] = field(default_factory=list)
    error_message: Optional[str] = None

    foreground_pixels: int = 0
    foreground_ratio: float = 0.0
    bbox: Optional[tuple[int, int, int, int]] = None
    centroid: Optional[tuple[float, float]] = None
    component_count: int = 0

    accepted: bool = True

    @property
    def is_reviewable(self) -> bool:
        """レビューで採否可能か (結果PNGがある完了フレーム)。"""
        return self.result_mask_path is not None and self.state in (
            FrameState.DONE, FrameState.WARNING,
        )


@dataclass
class PropagationSession:
    job_id: str
    state: PropagationUiState

    reference_entry_key: str
    reference_frame_index: int
    reference_mask_path: str

    order_mode: PropagationOrder
    direction: str

    frames: list[PropagationFrame]

    model_id: str
    precision: str
    device: str

    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    completed_count: int = 0
    failed_count: int = 0
    warning_count: int = 0

    # ------------------------------------------------------------------ #

    def frame_by_index(self, frame_index: int) -> Optional[PropagationFrame]:
        for f in self.frames:
            if f.frame_index == frame_index:
                return f
        return None

    def accepted_frames(self) -> list[PropagationFrame]:
        """採用かつ結果のある (基準を除く) フレーム。一括適用対象。"""
        return [
            f for f in self.frames
            if f.accepted and f.is_reviewable
            and f.frame_index != self.reference_frame_index
        ]

    def recompute_counts(self) -> None:
        self.completed_count = sum(
            1 for f in self.frames if f.state in (FrameState.DONE, FrameState.WARNING)
        )
        self.warning_count = sum(1 for f in self.frames if f.state == FrameState.WARNING)
        self.failed_count = sum(1 for f in self.frames if f.state == FrameState.FAILED)
