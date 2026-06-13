"""
GrabCut処理をGUIスレッドから分離するQObjectベースWorker (v0.4A.1)

使用方法:
  worker = GrabCutWorker(image, rect, options, request_id)
  thread = QThread()
  worker.moveToThread(thread)
  thread.started.connect(worker.run)
  worker.finished.connect(on_finished)
  worker.failed.connect(on_failed)
  thread.start()
"""

import logging
import traceback

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal

from core.grabcut_tool import GrabCutOptions, GrabCutResult, run_grabcut_optimized

_log = logging.getLogger(__name__)


class GrabCutWorker(QObject):
    """
    GrabCut処理をバックグラウンドスレッドで実行するWorker。
    Qtウィジェットを直接操作せず、Signalのみで通知する。
    """

    finished = Signal(object)   # GrabCutResult
    failed = Signal(str)        # ユーザー向けエラーメッセージ
    progress = Signal(str)      # 進捗メッセージ
    cancelled = Signal()

    def __init__(
        self,
        image_bgr: np.ndarray,
        rect: tuple[int, int, int, int],
        options: GrabCutOptions,
        request_id: int,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._image = image_bgr
        self._rect = rect
        self._options = options
        self._request_id = request_id
        self._cancel_requested = False

    @property
    def request_id(self) -> int:
        return self._request_id

    def request_cancel(self) -> None:
        """キャンセルフラグを立てる。cv2.grabCut() 実行前後で確認する。"""
        _log.info("キャンセル要求 (request_id=%d)", self._request_id)
        self._cancel_requested = True

    def run(self) -> None:
        """QThread の start() から呼ばれるエントリポイント。"""
        _log.info("GrabCutWorker 開始 (request_id=%d)", self._request_id)
        try:
            self.progress.emit("ROIを準備しています")
            if self._cancel_requested:
                _log.info("GrabCutキャンセル (ROI前, request_id=%d)", self._request_id)
                self.cancelled.emit()
                return

            if self._options.use_downscale:
                self.progress.emit(f"{self._options.max_processing_size}pxへ縮小しています")
            else:
                self.progress.emit("前景・背景を推定しています")

            if self._cancel_requested:
                _log.info("GrabCutキャンセル (GrabCut前, request_id=%d)", self._request_id)
                self.cancelled.emit()
                return

            result: GrabCutResult = run_grabcut_optimized(
                self._image, self._rect, self._options
            )

            if self._cancel_requested:
                _log.info("GrabCutキャンセル (GrabCut後, request_id=%d)", self._request_id)
                self.cancelled.emit()
                return

            self.progress.emit("結果を元解像度へ復元しています")
            _log.info(
                "GrabCutWorker 完了 (request_id=%d): 処理時間 %.3f秒, 縮小率 %.4f",
                self._request_id, result.processing_time_sec, result.scale,
            )
            self.finished.emit(result)

        except ValueError as e:
            _log.warning("GrabCut ValueError (request_id=%d): %s", self._request_id, e)
            self.failed.emit(str(e))
        except cv2.error as e:
            _log.error("GrabCut cv2.error (request_id=%d): %s", self._request_id, e)
            self.failed.emit(
                "OpenCV GrabCut処理に失敗しました。\n"
                "反復回数を下げるか、矩形範囲を変更してください。"
            )
        except MemoryError:
            _log.error("GrabCut MemoryError (request_id=%d)", self._request_id)
            self.failed.emit(
                "画像サイズが大きく、メモリを確保できませんでした。\n"
                "最大処理サイズを小さくしてください。"
            )
        except RuntimeError as e:
            _log.error("GrabCut RuntimeError (request_id=%d): %s", self._request_id, e)
            self.failed.emit(f"実行時エラーが発生しました: {e}")
        except Exception as e:
            _log.error(
                "GrabCut 予期しないエラー (request_id=%d):\n%s",
                self._request_id, traceback.format_exc(),
            )
            self.failed.emit(f"予期しないエラーが発生しました: {e}")
