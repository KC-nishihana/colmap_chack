"""
V0.8 Worker側: 全画像自動分割バッチを専用バックグラウンドスレッドで実行する。

メインスレッドは stdin を読み続けて pause/resume/cancel/status を Event 経由で伝える。
GPU ジョブは同時に 1 つだけ。各画像を独立処理し、1 枚の失敗が他画像へ波及しない。

このクラスは torch / sam2 を import しない。SAM 推論は注入された `generate_fn`
(本番: Sam2AmgManager.generate) を呼ぶだけなので、Fake generator で単体テストできる。

画像 1 枚の処理手順:
  キャッシュ有効確認 -> 有効なら skip -> 画像読込(RGB uint8) -> generate ->
  RLE検証つき決定的ソート(build_segment_arrays) -> NPZ一時保存+再読込検証+SHA256 ->
  manifest一時保存 -> batch_manifest更新 -> 次画像

stdout への JSON 書込は worker の Lock で保護した send_cb 経由で行う。
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

from ai import amg_cache, amg_manifest, amg_npz
from ai.amg_manifest import MANIFEST_NAME, SEGMENTS_NPZ_NAME
from ai.amg_protocol import (
    AmgErrorCode,
    AmgEvent,
    AmgImageStatus,
    make_job_error,
    make_job_event,
)
from ai.amg_rle import RleError

_log = logging.getLogger("sam_worker.amg_batch")


class _Cancelled(Exception):
    pass


class AmgBatchRunner(threading.Thread):
    def __init__(
        self,
        *,
        job_id: str,
        project_root: str,
        images: list[dict[str, Any]],          # [{"image_key","source_path"}, ...]
        settings: dict[str, Any],              # generator 設定 (preset 展開済み)
        preset: str,
        model: dict[str, Any],                 # model_id / sam2_commit / checkpoint_fingerprint
        generate_fn: Callable[[Any, dict], Any],   # (rgb, settings) -> result(.annotations,...)
        load_image_fn: Callable[[str], tuple],     # (path) -> (rgb, w, h)
        send_cb: Callable[[dict], None],
        force: bool = False,
        oom_retry: bool = True,
        on_finished: Optional[Callable[[str], None]] = None,
        reclaim_fn: Optional[Callable[[], None]] = None,
        reclaim_interval: int = 8,
    ) -> None:
        super().__init__(name=f"amg-{job_id}", daemon=True)
        self._job_id = job_id
        self._root = Path(project_root)
        self._images = list(images)
        self._settings = dict(settings)
        self._preset = preset
        self._model = dict(model)
        self._generate_fn = generate_fn
        self._load_image_fn = load_image_fn
        self._send = send_cb
        self._force = bool(force)
        self._oom_retry = bool(oom_retry)
        self._on_finished = on_finished
        self._reclaim_fn = reclaim_fn
        self._reclaim_interval = max(1, int(reclaim_interval))
        self._since_reclaim = 0

        self._pause = threading.Event()
        self._cancel = threading.Event()
        self._paused_emitted = False

        self._processed = 0
        self._succeeded = 0
        self._reused = 0
        self._failed = 0
        self._peak_vram_mb = 0

    # ---- 外部制御 ------------------------------------------------------ #

    def pause(self) -> None:
        self._pause.set()

    def resume(self) -> None:
        self._pause.clear()

    def cancel(self) -> None:
        self._cancel.set()
        self._pause.clear()

    @property
    def job_id(self) -> str:
        return self._job_id

    @property
    def total(self) -> int:
        return len(self._images)

    @property
    def processed(self) -> int:
        return self._processed

    @property
    def is_paused(self) -> bool:
        return self._pause.is_set()

    def status_snapshot(self) -> dict[str, int]:
        return {
            "processed": self._processed, "total": self.total,
            "succeeded": self._succeeded, "reused": self._reused,
            "failed": self._failed, "peak_vram_mb": self._peak_vram_mb,
        }

    # ---- スレッド本体 -------------------------------------------------- #

    def run(self) -> None:
        try:
            self._run_batch()
            self._emit(AmgEvent.BATCH_COMPLETED, **self.status_snapshot())
        except _Cancelled:
            self._emit(AmgEvent.BATCH_CANCELLED, **self.status_snapshot())
        except Exception as e:  # noqa: BLE001 — バッチ全体の想定外 (個別画像失敗は内部で処理)
            _log.error("[%s] バッチ失敗:\n%s", self._job_id, traceback.format_exc())
            self._send(make_job_error(AmgErrorCode.GENERATION_FAILED,
                                      f"バッチ処理に失敗しました: {e}", job_id=self._job_id))
            self._emit(AmgEvent.BATCH_FAILED, **self.status_snapshot())
        finally:
            if self._reclaim_fn is not None:
                try:
                    self._reclaim_fn()
                except Exception:
                    pass
            if self._on_finished is not None:
                self._on_finished(self._job_id)

    def _run_batch(self) -> None:
        batch_path = self._root / amg_manifest.CACHE_DIRNAME / amg_manifest.BATCH_MANIFEST_NAME
        for item in self._images:
            self._wait_if_paused()
            image_key = item["image_key"]
            source_path = item["source_path"]
            cache_id = amg_manifest.cache_id_for(image_key)
            cache_dir = self._root / amg_manifest.CACHE_DIRNAME / amg_manifest.IMAGES_DIRNAME / cache_id

            self._emit(AmgEvent.IMAGE_STARTED, image_key=image_key, cache_id=cache_id)
            try:
                reused = self._process_image(image_key, source_path, cache_dir, cache_id, batch_path)
            except _Cancelled:
                raise
            except Exception as e:  # 個別画像の失敗は分離して継続
                self._failed += 1
                self._processed += 1
                _log.error("[%s] 画像失敗 %s:\n%s", self._job_id, image_key, traceback.format_exc())
                amg_manifest.update_batch_image_entry(
                    batch_path, image_key, cache_id=cache_id,
                    status=AmgImageStatus.FAILED, error=str(e),
                )
                self._send(make_job_error(self._error_code(e), str(e),
                                          job_id=self._job_id, image_key=image_key))
                self._emit(AmgEvent.IMAGE_FAILED, image_key=image_key, cache_id=cache_id, error=str(e))
                self._emit_progress(image_key)
                self._maybe_reclaim(force=True)  # 失敗時は確実に回収 (OOM 後の断片化対策)
                continue

            self._processed += 1
            if reused:
                self._reused += 1
            else:
                self._succeeded += 1
                self._maybe_reclaim()  # 実解析した画像のみ周期的に VRAM/RAM を回収
            self._emit_progress(image_key)

    def _maybe_reclaim(self, force: bool = False) -> None:
        """一定枚数ごとに gc.collect + empty_cache で断片化を防ぐ (進むほど遅くなる対策)。"""
        if self._reclaim_fn is None:
            return
        self._since_reclaim += 1
        if force or self._since_reclaim >= self._reclaim_interval:
            self._since_reclaim = 0
            try:
                self._reclaim_fn()
            except Exception:
                pass

    # ---- 1 画像処理 ---------------------------------------------------- #

    def _process_image(self, image_key, source_path, cache_dir, cache_id, batch_path) -> bool:
        """戻り値: 再利用したら True (skip)、解析したら False。"""
        # 1. キャッシュ判定 (force でなければ。decode せず fingerprint で高速判定)
        if not self._force:
            chk = amg_cache.evaluate_cache(
                cache_dir, source_path=source_path,
                model=self._model, generator=self._settings,
            )
            if chk.state == amg_cache.REUSABLE:
                amg_manifest.update_batch_image_entry(
                    batch_path, image_key, cache_id=cache_id,
                    status=AmgImageStatus.READY,
                    segment_count=int((chk.manifest or {}).get("segment_count", 0)),
                    review_completed=bool((chk.manifest or {}).get("review", {}).get("completed", False)),
                )
                self._emit(AmgEvent.IMAGE_SKIPPED, image_key=image_key,
                           cache_id=cache_id, reason=chk.reason)
                return True

        # 2. processing マーク (異常終了検出用)
        amg_manifest.update_batch_image_entry(
            batch_path, image_key, cache_id=cache_id, status=AmgImageStatus.PROCESSING,
        )

        t0 = time.time()
        # 3. 画像読込 (RGB uint8)
        try:
            rgb, w, h = self._load_image_fn(source_path)
        except Exception as e:  # noqa: BLE001
            raise _ImageLoadError(str(e)) from e

        self._check_cancel()

        # 4. generate (注入された関数。OOM 再試行は manager 側)
        result = self._generate_fn(rgb, self._settings)
        annotations = getattr(result, "annotations", result)
        ppb_used = int(getattr(result, "points_per_batch_used", self._settings["points_per_batch"]))
        oom_retries = int(getattr(result, "oom_retries", 0))
        peak = int(getattr(result, "peak_vram_mb", 0))
        warnings = list(getattr(result, "warnings", []) or [])
        self._peak_vram_mb = max(self._peak_vram_mb, peak)
        # 大きな入力・結果ラッパは速やかに解放 (GPU/CPU メモリ肥大を防ぐ)
        del rgb, result

        self._check_cancel()

        # 5. RLE検証つき決定的ソート -> NPZ 配列
        try:
            arrays = amg_npz.build_segment_arrays(annotations, h, w)
        except RleError as e:
            raise _RleInvalid(str(e)) from e
        del annotations

        # 6. NPZ 原子保存 (内部で再読込検証) -> SHA256
        npz_path = cache_dir / SEGMENTS_NPZ_NAME
        try:
            sha = amg_npz.save_segments_npz(npz_path, arrays)
        except Exception as e:  # noqa: BLE001
            raise _ResultWriteError(f"NPZ 保存失敗: {e}") from e

        segment_ids = arrays["segment_ids"].tolist()
        segment_count = len(segment_ids)
        del arrays
        elapsed = time.time() - t0

        # generator block には実際に使用した points_per_batch を記録
        gen_for_manifest = dict(self._settings)
        gen_for_manifest["points_per_batch"] = ppb_used
        if oom_retries:
            warnings.append(f"OOM 再試行 {oom_retries} 回 (points_per_batch={ppb_used})")

        manifest = amg_manifest.build_image_manifest(
            image_key=image_key, source_path=source_path, width=w, height=h,
            model=self._model, generator=gen_for_manifest, preset=self._preset,
            segment_count=segment_count, segment_ids=segment_ids,
            segments_npz_sha256=sha, processing_time_sec=elapsed,
            status=AmgImageStatus.READY, warnings=warnings,
        )
        # settings_hash は元 settings(=要求値)で計算しキャッシュ判定と一致させる
        manifest["settings_hash"] = amg_manifest.settings_hash(self._settings, self._model)
        amg_manifest.atomic_write_json(cache_dir / MANIFEST_NAME, manifest)

        amg_manifest.update_batch_image_entry(
            batch_path, image_key, cache_id=cache_id,
            status=AmgImageStatus.READY, segment_count=segment_count,
        )
        self._emit(
            AmgEvent.IMAGE_COMPLETED, image_key=image_key, cache_id=cache_id,
            segment_count=segment_count, processing_time_sec=round(elapsed, 4),
            manifest_path=str(cache_dir / MANIFEST_NAME),
            points_per_batch_used=ppb_used, oom_retries=oom_retries,
        )
        return False

    # ---- pause / cancel / progress ------------------------------------ #

    def _check_cancel(self) -> None:
        if self._cancel.is_set():
            raise _Cancelled()

    def _wait_if_paused(self) -> None:
        self._check_cancel()
        if self._pause.is_set():
            if not self._paused_emitted:
                self._emit(AmgEvent.BATCH_PAUSED, **self.status_snapshot())
                self._paused_emitted = True
            while self._pause.is_set():
                if self._cancel.is_set():
                    raise _Cancelled()
                time.sleep(0.05)
            if self._paused_emitted:
                self._emit(AmgEvent.BATCH_RESUMED, **self.status_snapshot())
                self._paused_emitted = False

    def _emit_progress(self, image_key) -> None:
        self._emit(AmgEvent.BATCH_PROGRESS, current_image=image_key, **self.status_snapshot())

    def _emit(self, event: str, **fields) -> None:
        self._send(make_job_event(event, self._job_id, **fields))

    @staticmethod
    def _error_code(e: Exception) -> str:
        if isinstance(e, _ImageLoadError):
            return AmgErrorCode.IMAGE_LOAD_FAILED
        if isinstance(e, _RleInvalid):
            return AmgErrorCode.RLE_INVALID
        if isinstance(e, _ResultWriteError):
            return AmgErrorCode.RESULT_WRITE_FAILED
        if type(e).__name__ == "AmgOom":
            return AmgErrorCode.GPU_OOM
        return AmgErrorCode.GENERATION_FAILED


class _ImageLoadError(Exception):
    pass


class _RleInvalid(Exception):
    pass


class _ResultWriteError(Exception):
    pass
