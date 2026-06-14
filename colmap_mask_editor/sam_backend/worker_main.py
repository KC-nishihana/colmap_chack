"""
SAM 2.1 Worker のエントリポイント (QProcess 子プロセス)。

  python -u worker_main.py

で起動され、stdin から JSON Lines コマンドを受け、stdout へ JSON Lines 応答を返す。
ログ・警告・トレースバックは stderr / ログファイルへ (stdout は JSON 専用)。

torch / sam2 は load_model 等で実際に必要になるまで import しない。
hello / health は torch 未導入でも応答を返す (能力を報告し AI を無効化させる)。
"""

from __future__ import annotations

import json
import sys
import threading
import uuid
from pathlib import Path

# --- stdout 汚染防止 ---------------------------------------------------------
# 一部ライブラリは import 時に print する。stdout は JSON 専用にするため、
# プロトコル用に本物の stdout を退避し、sys.stdout は stderr へ向ける。
_REAL_STDOUT = sys.stdout
sys.stdout = sys.stderr

# パッケージルートを import パスへ (PYTHONPATH が無い直接起動でも動くように)
_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from ai import protocol  # noqa: E402
from ai import model_registry  # noqa: E402
from ai.propagation_protocol import (  # noqa: E402
    PropagationCommand,
    PropagationDirection,
    PropagationErrorCode,
    PropagationEvent,
    make_job_error,
    make_job_event,
)
from ai.runtime_paths import get_propagation_job_dir  # noqa: E402
from sam_backend import result_writer  # noqa: E402
from sam_backend.image_loader import ImageLoadError, load_image_rgb  # noqa: E402
from sam_backend.worker_logging import setup_worker_logging  # noqa: E402

_log = setup_worker_logging()

_MANIFEST_PATH = Path(__file__).resolve().parent / "sam2_manifest.json"


def _read_sam2_commit() -> str:
    try:
        with open(_MANIFEST_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return str(data.get("commit", ""))
    except Exception:
        return ""


class Worker:
    def __init__(self) -> None:
        self._model_manager = None     # 遅延生成 (torch import を伴うため)
        self._predictor = None
        self._sam2_commit = _read_sam2_commit()
        self._running = True
        # V0.7 伝播: stdout は複数スレッドから書かれるため Lock で保護する
        self._stdout_lock = threading.Lock()
        self._video_manager = None
        self._prop_runner = None       # 実行中の PropagationRunner (1ジョブのみ)

    # ------------------------------------------------------------------ #
    # 送信
    # ------------------------------------------------------------------ #

    def _send(self, msg: dict) -> None:
        line = protocol.encode_line(msg)
        with self._stdout_lock:
            _REAL_STDOUT.write(line + "\n")
            _REAL_STDOUT.flush()

    def _gpu_busy(self) -> bool:
        return self._prop_runner is not None and self._prop_runner.is_alive()

    def _reject_if_busy(self, request_id) -> bool:
        """伝播中は単一画像系のGPUコマンドを拒否する。拒否したら True。"""
        if self._gpu_busy():
            self._send(make_job_error(
                PropagationErrorCode.BUSY,
                "画像伝播を実行中のため、この操作は実行できません。",
                job_id=self._prop_runner.job_id if self._prop_runner else None,
                request_id=request_id,
            ))
            return True
        return False

    def _send_error(self, error_code: str, message: str, request_id=None, **fields) -> None:
        self._send(protocol.make_error(error_code, message, request_id, **fields))

    # ------------------------------------------------------------------ #
    # メインループ
    # ------------------------------------------------------------------ #

    def run(self) -> int:
        _log.info("SAM Worker 起動 (python=%s)", sys.executable)
        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except (ValueError, TypeError) as e:
                self._send_error(protocol.ErrorCode.BAD_REQUEST, f"不正なJSON: {e}")
                continue
            if not isinstance(msg, dict):
                self._send_error(protocol.ErrorCode.BAD_REQUEST, "JSONがオブジェクトではありません")
                continue

            try:
                self._dispatch(msg)
            except Exception as e:  # 想定外。Worker は落とさずエラー応答。
                import traceback
                _log.error("コマンド処理中の例外:\n%s", traceback.format_exc())
                self._send_error(
                    protocol.ErrorCode.INTERNAL,
                    f"内部エラー: {e}",
                    request_id=msg.get("request_id"),
                )

            if not self._running:
                break

        _log.info("SAM Worker 終了")
        return 0

    def _dispatch(self, msg: dict) -> None:
        command = msg.get("command")
        request_id = msg.get("request_id")

        if command == protocol.Command.HELLO:
            self._cmd_hello(request_id)
        elif command == protocol.Command.HEALTH:
            self._cmd_health(request_id)
        elif command == protocol.Command.LOAD_MODEL:
            self._cmd_load_model(msg, request_id)
        elif command == protocol.Command.UNLOAD_MODEL:
            self._cmd_unload_model(request_id)
        elif command == protocol.Command.SET_IMAGE:
            self._cmd_set_image(msg, request_id)
        elif command == protocol.Command.PREDICT:
            self._cmd_predict(msg, request_id)
        elif command == protocol.Command.RELEASE_IMAGE:
            self._cmd_release_image(msg, request_id)
        elif command == protocol.Command.CLEAR_CUDA_CACHE:
            self._cmd_clear_cuda_cache(request_id)
        elif command == protocol.Command.SHUTDOWN:
            self._cmd_shutdown(request_id)
        elif command == PropagationCommand.START:
            self._cmd_propagation_start(msg, request_id)
        elif command == PropagationCommand.PAUSE:
            self._cmd_propagation_control(msg, request_id, "pause")
        elif command == PropagationCommand.RESUME:
            self._cmd_propagation_control(msg, request_id, "resume")
        elif command == PropagationCommand.CANCEL:
            self._cmd_propagation_control(msg, request_id, "cancel")
        elif command == PropagationCommand.STATUS:
            self._cmd_propagation_status(msg, request_id)
        elif command == PropagationCommand.RELEASE:
            self._cmd_propagation_release(msg, request_id)
        else:
            self._send_error(
                protocol.ErrorCode.BAD_REQUEST,
                f"不明なコマンド: {command!r}", request_id,
            )

    # ------------------------------------------------------------------ #
    # 能力チェック (hello / health 共通)
    # ------------------------------------------------------------------ #

    def _probe(self):
        from sam_backend.sam2_model_manager import probe_environment
        return probe_environment()

    def _cmd_hello(self, request_id) -> None:
        caps = self._probe()
        payload = {
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "torch_version": caps.torch_version,
            "torch_cuda_version": caps.torch_cuda_version,
            "torchvision_version": caps.torchvision_version,
            "cuda_available": caps.cuda_available,
            "cuda_extension_loaded": caps.cuda_extension_loaded,
            "gpu_name": caps.gpu_name,
            "compute_capability": caps.compute_capability,
            "sam2_commit": self._sam2_commit,
        }
        if caps.torch_import_error:
            payload["torch_import_error"] = caps.torch_import_error
        if caps.sam2_import_error:
            payload["sam2_import_error"] = caps.sam2_import_error
        if caps.cuda_extension_error:
            payload["cuda_extension_error"] = caps.cuda_extension_error
        if not caps.cuda_extension_loaded:
            payload["message"] = (
                "SAM 2 CUDA拡張を読み込めませんでした。AIセグメンテーションは使用できません。"
            )
        self._send(protocol.make_event(protocol.Event.READY, request_id, **payload))

    def _cmd_health(self, request_id) -> None:
        caps = self._probe()
        if not caps.cuda_extension_loaded:
            self._send_error(
                protocol.ErrorCode.CUDA_EXTENSION_UNAVAILABLE,
                "SAM 2 CUDA拡張を読み込めませんでした。AI機能は利用できません。",
                request_id,
                cuda_available=caps.cuda_available,
            )
            return
        self._send(protocol.make_event(
            protocol.Event.HEALTH_RESULT, request_id,
            cuda_available=caps.cuda_available,
            cuda_extension_loaded=caps.cuda_extension_loaded,
            model_loaded=(self._model_manager is not None and self._model_manager.is_loaded),
            vram_allocated_mb=(self._model_manager.vram_allocated_mb() if self._model_manager else 0),
        ))

    # ------------------------------------------------------------------ #
    # モデル
    # ------------------------------------------------------------------ #

    def _ensure_manager(self):
        if self._model_manager is None:
            from sam_backend.sam2_model_manager import Sam2ModelManager
            self._model_manager = Sam2ModelManager()
        return self._model_manager

    def _cmd_load_model(self, msg: dict, request_id) -> None:
        if self._reject_if_busy(request_id):
            return
        from sam_backend.sam2_model_manager import (
            CudaExtensionUnavailable,
            CudaUnavailable,
            PrecisionUnavailable,
        )

        model_id = msg.get("model_id", model_registry.DEFAULT_MODEL_ID)
        checkpoint_path = msg.get("checkpoint_path")
        precision = msg.get("precision", "bf16")
        device = msg.get("device", "cuda:0")

        if not model_registry.has_model(model_id):
            self._send_error(protocol.ErrorCode.BAD_REQUEST,
                             f"未登録のモデル: {model_id}", request_id)
            return
        info = model_registry.get_model(model_id)

        if not checkpoint_path or not Path(checkpoint_path).exists():
            self._send_error(
                protocol.ErrorCode.MODEL_FILE_NOT_FOUND,
                f"チェックポイントが見つかりません: {checkpoint_path}", request_id,
            )
            return

        mm = self._ensure_manager()
        try:
            res = mm.load(
                model_id=model_id,
                config_name=info.config_name,
                checkpoint_path=str(checkpoint_path),
                precision=precision,
                device=device,
            )
        except CudaExtensionUnavailable as e:
            self._send_error(protocol.ErrorCode.CUDA_EXTENSION_UNAVAILABLE, str(e), request_id)
            return
        except CudaUnavailable as e:
            self._send_error(protocol.ErrorCode.CUDA_UNAVAILABLE, str(e), request_id)
            return
        except PrecisionUnavailable as e:
            self._send_error(protocol.ErrorCode.PRECISION_UNAVAILABLE, str(e), request_id)
            return
        except FileNotFoundError as e:
            self._send_error(protocol.ErrorCode.MODEL_CONFIG_NOT_FOUND, str(e), request_id)
            return
        except Exception as e:
            self._send_error(protocol.ErrorCode.MODEL_LOAD_FAILED, f"モデルロード失敗: {e}", request_id)
            return

        from sam_backend.sam2_predictor import Sam2Predictor
        self._predictor = Sam2Predictor(mm)
        self._send(protocol.make_event(
            protocol.Event.MODEL_LOADED, request_id,
            model_id=model_id, device=device, precision=precision,
            vram_allocated_mb=res["vram_allocated_mb"],
        ))

    def _cmd_unload_model(self, request_id) -> None:
        if self._reject_if_busy(request_id):
            return
        if self._model_manager is not None:
            self._model_manager.unload()
        self._predictor = None
        self._send(protocol.make_event(protocol.Event.MODEL_UNLOADED, request_id))

    # ------------------------------------------------------------------ #
    # 画像
    # ------------------------------------------------------------------ #

    def _cmd_set_image(self, msg: dict, request_id) -> None:
        if self._reject_if_busy(request_id):
            return
        if self._predictor is None:
            self._send_error(protocol.ErrorCode.MODEL_NOT_LOADED,
                             "モデルがロードされていません", request_id)
            return
        image_path = msg.get("image_path")
        try:
            rgb, w, h = load_image_rgb(image_path)
        except ImageLoadError as e:
            code = (protocol.ErrorCode.IMAGE_NOT_FOUND
                    if "存在しません" in str(e) else protocol.ErrorCode.IMAGE_LOAD_FAILED)
            self._send_error(code, str(e), request_id)
            return

        image_key = uuid.uuid4().hex
        try:
            elapsed = self._predictor.set_image(rgb, image_key)
        except Exception as e:
            if self._is_oom(e):
                self._handle_oom()
                self._send_error(protocol.ErrorCode.CUDA_OOM,
                                 "CUDA メモリ不足で画像Embeddingに失敗しました", request_id)
                return
            self._send_error(protocol.ErrorCode.IMAGE_LOAD_FAILED,
                             f"Embedding生成に失敗しました: {e}", request_id)
            return

        self._send(protocol.make_event(
            protocol.Event.IMAGE_READY, request_id,
            width=w, height=h, embedding_time_sec=round(elapsed, 4), image_key=image_key,
        ))

    def _cmd_release_image(self, msg: dict, request_id) -> None:
        if self._predictor is not None:
            self._predictor.release()
        self._send(protocol.make_event(protocol.Event.IMAGE_RELEASED, request_id))

    # ------------------------------------------------------------------ #
    # 推論
    # ------------------------------------------------------------------ #

    def _cmd_predict(self, msg: dict, request_id) -> None:
        if self._reject_if_busy(request_id):
            return
        if self._predictor is None:
            self._send_error(protocol.ErrorCode.MODEL_NOT_LOADED,
                             "モデルがロードされていません", request_id)
            return

        image_key = msg.get("image_key")
        if image_key != self._predictor.image_key:
            self._send_error(
                protocol.ErrorCode.IMAGE_KEY_MISMATCH,
                "画像が切り替わっています。画像を再設定してください。", request_id,
            )
            return

        points = msg.get("points", []) or []
        box = msg.get("box")
        multimask = bool(msg.get("multimask_output", True))

        if not points and box is None:
            self._send_error(protocol.ErrorCode.BAD_REQUEST,
                             "プロンプトがありません", request_id)
            return

        try:
            masks, scores, elapsed = self._predictor.predict(points, box, multimask)
        except Exception as e:
            if self._is_oom(e):
                self._handle_oom()
                self._send_error(protocol.ErrorCode.CUDA_OOM,
                                 "CUDA メモリ不足で推論に失敗しました", request_id)
                return
            import traceback
            _log.error("推論失敗:\n%s", traceback.format_exc())
            self._send_error(protocol.ErrorCode.PREDICT_FAILED, f"推論に失敗しました: {e}", request_id)
            return

        h, w = (masks.shape[1], masks.shape[2]) if masks.ndim == 3 else masks.shape
        try:
            result_path = result_writer.write_result_npz(
                masks=masks, scores=scores, request_id=int(request_id), image_key=image_key,
            )
        except Exception as e:
            self._send_error(protocol.ErrorCode.INTERNAL,
                             f"結果ファイルの書き出しに失敗しました: {e}", request_id)
            return

        vram = self._model_manager.vram_allocated_mb() if self._model_manager else 0
        self._send(protocol.make_event(
            protocol.Event.PREDICTION_READY, request_id,
            image_key=image_key,
            result_path=result_path,
            mask_count=int(masks.shape[0]),
            scores=[round(float(s), 4) for s in scores.tolist()],
            width=int(w), height=int(h),
            prediction_time_sec=round(elapsed, 4),
            vram_allocated_mb=vram,
        ))

    # ------------------------------------------------------------------ #
    # その他
    # ------------------------------------------------------------------ #

    def _cmd_clear_cuda_cache(self, request_id) -> None:
        if self._model_manager is not None:
            self._model_manager.clear_cuda_cache()
        self._send(protocol.make_event(protocol.Event.CUDA_CACHE_CLEARED, request_id))

    # ------------------------------------------------------------------ #
    # 伝播 (V0.7) — GPUジョブはバックグラウンドスレッドで実行
    # ------------------------------------------------------------------ #

    def _cmd_propagation_start(self, msg: dict, request_id) -> None:
        if self._gpu_busy():
            self._send(make_job_error(PropagationErrorCode.BUSY,
                                      "別の伝播ジョブが実行中です", request_id=request_id))
            return

        frames = msg.get("frames") or []
        if len(frames) < 2:
            self._send(make_job_error(PropagationErrorCode.INVALID_SEQUENCE,
                                      "伝播対象が2枚未満です", request_id=request_id))
            return
        ref_idx = msg.get("reference_frame_index")
        if not isinstance(ref_idx, int) or not (0 <= ref_idx < len(frames)):
            self._send(make_job_error(PropagationErrorCode.INVALID_SEQUENCE,
                                      "基準フレーム位置が不正です", request_id=request_id))
            return
        ref_mask_path = msg.get("reference_mask_path")
        if not ref_mask_path or not Path(ref_mask_path).exists():
            self._send(make_job_error(PropagationErrorCode.INVALID_REFERENCE_MASK,
                                      f"基準マスクが見つかりません: {ref_mask_path}",
                                      request_id=request_id))
            return
        model_id = msg.get("model_id", model_registry.DEFAULT_MODEL_ID)
        if not model_registry.has_model(model_id):
            self._send(protocol.make_error(protocol.ErrorCode.BAD_REQUEST,
                                           f"未登録モデル: {model_id}", request_id))
            return
        checkpoint_path = msg.get("checkpoint_path")
        if not checkpoint_path or not Path(checkpoint_path).exists():
            self._send(protocol.make_error(protocol.ErrorCode.MODEL_FILE_NOT_FOUND,
                                           f"チェックポイントが見つかりません: {checkpoint_path}",
                                           request_id))
            return
        direction = msg.get("direction", PropagationDirection.BOTH)
        if direction not in PropagationDirection.ALL:
            self._send(make_job_error(PropagationErrorCode.INVALID_SEQUENCE,
                                      f"不正な伝播方向: {direction}", request_id=request_id))
            return

        info = model_registry.get_model(model_id)
        job_id = "prop-" + uuid.uuid4().hex[:10]
        job_dir = get_propagation_job_dir(job_id)

        # VRAM確保のため単一画像Embeddingを解放 (モデルは保持)
        try:
            if self._predictor is not None:
                self._predictor.release()
        except Exception:
            pass

        from sam_backend.propagation_runner import PropagationRunner
        from sam_backend.sam2_video_manager import Sam2VideoManager

        self._video_manager = Sam2VideoManager()
        params = {
            "config_name": info.config_name,
            "checkpoint_path": str(checkpoint_path),
            "model_id": model_id,
            "precision": msg.get("precision", "bf16"),
            "device": msg.get("device", "cuda:0"),
            "frames": frames,
            "reference_frame_index": ref_idx,
            "reference_mask_path": str(ref_mask_path),
            "direction": direction,
            "offload_video_to_cpu": bool(msg.get("offload_video_to_cpu", True)),
            "offload_state_to_cpu": bool(msg.get("offload_state_to_cpu", False)),
            "async_loading_frames": bool(msg.get("async_loading_frames", False)),
            "max_frames": int(msg.get("max_frames", 100)),
            "jpeg_quality": int(msg.get("jpeg_quality", 95)),
            "thresholds": msg.get("thresholds"),
        }
        runner = PropagationRunner(
            job_id=job_id, params=params, video_manager=self._video_manager,
            job_dir=job_dir, send_cb=self._send, on_finished=self._on_prop_finished,
        )
        self._prop_runner = runner
        # 受付応答: ここで request_id を解決し、以後は job_id で進捗を追う
        self._send(make_job_event(PropagationEvent.STARTED, job_id,
                                  request_id=request_id, frame_count=len(frames)))
        runner.start()

    def _on_prop_finished(self, job_id: str) -> None:
        r = self._prop_runner
        if r is not None and r.job_id == job_id:
            self._prop_runner = None

    def _cmd_propagation_control(self, msg: dict, request_id, action: str) -> None:
        r = self._prop_runner
        job_id = msg.get("job_id")
        if r is None or not r.is_alive() or (job_id and job_id != r.job_id):
            self._send(make_job_error(PropagationErrorCode.NOT_FOUND,
                                      "対象の伝播ジョブが見つかりません",
                                      job_id=job_id, request_id=request_id))
            return
        if action == "pause":
            r.pause()
            self._send(make_job_event(PropagationEvent.PAUSED, r.job_id, request_id=request_id))
        elif action == "resume":
            r.resume()
            self._send(make_job_event(PropagationEvent.RESUMED, r.job_id, request_id=request_id))
        elif action == "cancel":
            r.cancel()
            self._send(make_job_event(PropagationEvent.CANCELLING, r.job_id, request_id=request_id))

    def _cmd_propagation_status(self, msg: dict, request_id) -> None:
        r = self._prop_runner
        if r is None or not r.is_alive():
            self._send(make_job_error(PropagationErrorCode.NOT_FOUND,
                                      "実行中の伝播ジョブはありません", request_id=request_id))
            return
        self._send(make_job_event(PropagationEvent.PROGRESS, r.job_id, request_id=request_id,
                                  processed=r.processed, total=r.total, paused=r.is_paused))

    def _cmd_propagation_release(self, msg: dict, request_id) -> None:
        job_id = msg.get("job_id")
        r = self._prop_runner
        if r is not None and r.is_alive():
            r.cancel()
            r.join(timeout=10.0)
        if self._video_manager is not None:
            try:
                self._video_manager.release()
            except Exception:
                pass
            self._video_manager = None
        self._prop_runner = None
        self._send(make_job_event(PropagationEvent.RELEASED,
                                  job_id or (r.job_id if r else ""), request_id=request_id))

    def _cmd_shutdown(self, request_id) -> None:
        self._send(protocol.make_event(protocol.Event.SHUTTING_DOWN, request_id))
        r = self._prop_runner
        if r is not None and r.is_alive():
            r.cancel()
            r.join(timeout=10.0)
        if self._video_manager is not None:
            try:
                self._video_manager.release()
            except Exception:
                pass
        if self._model_manager is not None:
            self._model_manager.unload()
        self._running = False

    # ------------------------------------------------------------------ #
    # OOM ハンドリング
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_oom(exc: Exception) -> bool:
        name = type(exc).__name__
        if name == "OutOfMemoryError":
            return True
        return "out of memory" in str(exc).lower()

    def _handle_oom(self) -> None:
        _log.error("CUDA OOM を検出。Embedding解放 + cache クリア。")
        try:
            if self._predictor is not None:
                self._predictor.release()
            if self._model_manager is not None:
                self._model_manager.clear_cuda_cache()
        except Exception:
            pass


def main() -> int:
    return Worker().run()


if __name__ == "__main__":
    sys.exit(main())
