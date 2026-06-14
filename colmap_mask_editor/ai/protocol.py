"""
GUI (QProcess親) と SAM Worker (子プロセス) の通信プロトコル。

設計方針 (CLAUDE.md V0.6 要件):
  - stdout は JSON Lines 専用。1行 = 1つの JSON オブジェクト。UTF-8。
  - ログ・警告・トレースバックは stderr / ログファイルへ (stdout へ出さない)。
  - マスク本体は JSON へ埋め込まず、NPZ一時ファイルのパスのみ受け渡す。
  - protocol_version で互換性を確認する。

このモジュールは純粋ロジックのみ。torch / sam2 / PySide6 に依存しない
(GUI・Worker・テストのいずれからも import できる)。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

PROTOCOL_VERSION = 1

# ------------------------------------------------------------------ #
# コマンド (GUI -> Worker) / イベント (Worker -> GUI) の名前定数
# ------------------------------------------------------------------ #


class Command:
    HELLO = "hello"
    HEALTH = "health"
    LOAD_MODEL = "load_model"
    UNLOAD_MODEL = "unload_model"
    SET_IMAGE = "set_image"
    PREDICT = "predict"
    RELEASE_IMAGE = "release_image"
    CLEAR_CUDA_CACHE = "clear_cuda_cache"
    SHUTDOWN = "shutdown"

    ALL = frozenset({
        HELLO, HEALTH, LOAD_MODEL, UNLOAD_MODEL, SET_IMAGE,
        PREDICT, RELEASE_IMAGE, CLEAR_CUDA_CACHE, SHUTDOWN,
    })


class Event:
    READY = "ready"
    HEALTH_RESULT = "health_result"
    MODEL_LOADED = "model_loaded"
    MODEL_UNLOADED = "model_unloaded"
    IMAGE_READY = "image_ready"
    IMAGE_RELEASED = "image_released"
    PREDICTION_READY = "prediction_ready"
    CUDA_CACHE_CLEARED = "cuda_cache_cleared"
    SHUTTING_DOWN = "shutting_down"
    ERROR = "error"
    LOG = "log"   # 構造化ログを stdout 経由で送りたい場合 (任意)


class Status:
    OK = "ok"
    ERROR = "error"


class ErrorCode:
    """error イベントの error_code。GUI 側で分岐表示するために使う。"""
    CUDA_EXTENSION_UNAVAILABLE = "CUDA_EXTENSION_UNAVAILABLE"
    CUDA_UNAVAILABLE = "CUDA_UNAVAILABLE"
    TORCH_IMPORT_FAILED = "TORCH_IMPORT_FAILED"
    SAM2_IMPORT_FAILED = "SAM2_IMPORT_FAILED"
    MODEL_NOT_LOADED = "MODEL_NOT_LOADED"
    MODEL_FILE_NOT_FOUND = "MODEL_FILE_NOT_FOUND"
    MODEL_CONFIG_NOT_FOUND = "MODEL_CONFIG_NOT_FOUND"
    MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"
    IMAGE_NOT_FOUND = "IMAGE_NOT_FOUND"
    IMAGE_LOAD_FAILED = "IMAGE_LOAD_FAILED"
    IMAGE_KEY_MISMATCH = "IMAGE_KEY_MISMATCH"
    PREDICT_FAILED = "PREDICT_FAILED"
    CUDA_OOM = "CUDA_OOM"
    BAD_REQUEST = "BAD_REQUEST"
    PRECISION_UNAVAILABLE = "PRECISION_UNAVAILABLE"
    INTERNAL = "INTERNAL"


# ------------------------------------------------------------------ #
# メッセージ生成ヘルパー (Worker 側で応答を作る / GUI 側で要求を作る)
# ------------------------------------------------------------------ #


def make_request(command: str, request_id: int, **fields: Any) -> dict[str, Any]:
    """GUI -> Worker のコマンド辞書を作る。"""
    msg: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": int(request_id),
        "command": command,
    }
    msg.update(fields)
    return msg


def make_event(event: str, request_id: Optional[int] = None, **fields: Any) -> dict[str, Any]:
    """Worker -> GUI の成功イベント辞書を作る。"""
    msg: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "status": Status.OK,
        "event": event,
    }
    if request_id is not None:
        msg["request_id"] = int(request_id)
    msg.update(fields)
    return msg


def make_error(
    error_code: str,
    message: str,
    request_id: Optional[int] = None,
    **fields: Any,
) -> dict[str, Any]:
    """Worker -> GUI のエラー辞書を作る。"""
    msg: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "status": Status.ERROR,
        "event": Event.ERROR,
        "error_code": error_code,
        "message": message,
    }
    if request_id is not None:
        msg["request_id"] = int(request_id)
    msg.update(fields)
    return msg


def encode_message(msg: dict[str, Any]) -> bytes:
    """辞書を JSON 1行 (改行終端・UTF-8) へエンコードする。

    日本語パスを含むため ensure_ascii=False。NPZ等のバイナリは含めない前提。
    """
    line = json.dumps(msg, ensure_ascii=False, separators=(",", ":"))
    return (line + "\n").encode("utf-8")


def encode_line(msg: dict[str, Any]) -> str:
    """辞書を JSON 1行文字列へエンコードする (Worker stdout への print 用)。"""
    return json.dumps(msg, ensure_ascii=False, separators=(",", ":"))


# ------------------------------------------------------------------ #
# 受信側: ストリームを行単位の JSON へ復元するデコーダ
# ------------------------------------------------------------------ #


@dataclass
class ParsedLine:
    """1行分のデコード結果。"""
    ok: bool
    obj: Optional[dict[str, Any]] = None
    raw: str = ""
    error: str = ""


@dataclass
class JsonLineDecoder:
    """
    QProcess の readyReadStandardOutput では、1回の read で

      - 1行の途中までしか来ない (分割)
      - 複数行まとめて来る
      - 不正な (JSON でない) 行が混じる

    いずれも起こりうる。feed() にバイト/文字列チャンクを渡すと、完成した
    行だけを ParsedLine のリストで返し、未完成分は内部バッファへ残す。

    不正な行は ok=False の ParsedLine として返し (例外は投げない)、
    呼び出し側が破棄やログ出力を判断できるようにする。
    """

    _buffer: str = field(default="", repr=False)
    max_line_bytes: int = 64 * 1024 * 1024  # 暴走時の保険 (通常はパスのみで十分小さい)

    def feed(self, data) -> list[ParsedLine]:
        if isinstance(data, (bytes, bytearray)):
            text = bytes(data).decode("utf-8", errors="replace")
        else:
            text = str(data)

        self._buffer += text
        results: list[ParsedLine] = []

        # 行区切りは \n。CR は除去して CRLF / LF どちらも扱う。
        while True:
            idx = self._buffer.find("\n")
            if idx < 0:
                break
            line = self._buffer[:idx]
            self._buffer = self._buffer[idx + 1:]
            line = line.rstrip("\r")
            if line.strip() == "":
                continue
            results.append(self._parse_line(line))

        # バッファ暴走防止
        if len(self._buffer.encode("utf-8", errors="ignore")) > self.max_line_bytes:
            bad = self._buffer
            self._buffer = ""
            results.append(ParsedLine(ok=False, raw=bad, error="行が長すぎます (区切り欠落の疑い)"))

        return results

    @staticmethod
    def _parse_line(line: str) -> ParsedLine:
        try:
            obj = json.loads(line)
        except (ValueError, TypeError) as e:
            return ParsedLine(ok=False, raw=line, error=f"JSON解析失敗: {e}")
        if not isinstance(obj, dict):
            return ParsedLine(ok=False, raw=line, error="JSONがオブジェクトではありません")
        return ParsedLine(ok=True, obj=obj, raw=line)

    def reset(self) -> None:
        self._buffer = ""

    @property
    def pending(self) -> str:
        """未完成のバッファ内容 (デバッグ用)。"""
        return self._buffer


# ------------------------------------------------------------------ #
# 検証ヘルパー
# ------------------------------------------------------------------ #


def is_error(msg: dict[str, Any]) -> bool:
    return msg.get("status") == Status.ERROR or msg.get("event") == Event.ERROR


def protocol_matches(msg: dict[str, Any]) -> bool:
    return msg.get("protocol_version") == PROTOCOL_VERSION
