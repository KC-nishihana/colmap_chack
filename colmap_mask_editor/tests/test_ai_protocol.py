"""
JSON Lines プロトコルのテスト (torch / GPU / QProcess 不要)。

確認:
  - 分割された stdout を正しく1行JSONへ復元
  - 1回の feed で複数JSONを処理
  - 不正JSONを破棄 (例外を投げない)
  - 日本語/全角スペースを含むパスを往復できる
"""

import json

from ai import protocol
from ai.protocol import JsonLineDecoder


def test_make_request_has_protocol_version():
    msg = protocol.make_request(protocol.Command.PREDICT, 5, image_key="k")
    assert msg["protocol_version"] == protocol.PROTOCOL_VERSION
    assert msg["request_id"] == 5
    assert msg["command"] == "predict"
    assert msg["image_key"] == "k"


def test_make_error_shape():
    msg = protocol.make_error(protocol.ErrorCode.CUDA_OOM, "メモリ不足", 7)
    assert protocol.is_error(msg)
    assert msg["error_code"] == "CUDA_OOM"
    assert msg["request_id"] == 7


def test_encode_decode_roundtrip():
    msg = protocol.make_event(protocol.Event.IMAGE_READY, 3, image_key="abc", width=100)
    data = protocol.encode_message(msg)
    assert data.endswith(b"\n")
    dec = JsonLineDecoder()
    out = dec.feed(data)
    assert len(out) == 1
    assert out[0].ok
    assert out[0].obj["image_key"] == "abc"


def test_decoder_split_across_chunks():
    """1行のJSONが複数チャンクに分割されても復元できる。"""
    msg = protocol.make_event(protocol.Event.READY, 1, gpu_name="RTX 4090")
    data = protocol.encode_message(msg)
    mid = len(data) // 2
    dec = JsonLineDecoder()
    out1 = dec.feed(data[:mid])
    assert out1 == []  # まだ行が完成していない
    out2 = dec.feed(data[mid:])
    assert len(out2) == 1
    assert out2[0].obj["gpu_name"] == "RTX 4090"


def test_decoder_multiple_lines_in_one_read():
    """1回の read で複数JSON行が来ても全て処理する。"""
    m1 = protocol.encode_message(protocol.make_event("a", 1))
    m2 = protocol.encode_message(protocol.make_event("b", 2))
    m3 = protocol.encode_message(protocol.make_event("c", 3))
    dec = JsonLineDecoder()
    out = dec.feed(m1 + m2 + m3)
    assert len(out) == 3
    assert [o.obj["event"] for o in out] == ["a", "b", "c"]


def test_decoder_invalid_json_does_not_raise():
    dec = JsonLineDecoder()
    out = dec.feed(b"not json at all\n")
    assert len(out) == 1
    assert out[0].ok is False
    assert out[0].raw == "not json at all"


def test_decoder_mixed_valid_and_invalid():
    valid = protocol.encode_message(protocol.make_event("ok", 1))
    dec = JsonLineDecoder()
    out = dec.feed(b"garbage\n" + valid)
    assert len(out) == 2
    assert out[0].ok is False
    assert out[1].ok is True


def test_decoder_handles_crlf():
    dec = JsonLineDecoder()
    line = json.dumps({"event": "x", "protocol_version": 1})
    out = dec.feed((line + "\r\n").encode("utf-8"))
    assert len(out) == 1
    assert out[0].obj["event"] == "x"


def test_decoder_non_object_json_is_error():
    dec = JsonLineDecoder()
    out = dec.feed(b"[1, 2, 3]\n")
    assert out[0].ok is False


def test_unicode_path_roundtrip():
    """日本語・全角スペースを含むパスを JSON で往復できる。"""
    path = "C:/プロジェクト/images/画像　001.jpg"  # 全角スペース
    msg = protocol.make_request(protocol.Command.SET_IMAGE, 1, image_path=path)
    data = protocol.encode_message(msg)
    dec = JsonLineDecoder()
    out = dec.feed(data)
    assert out[0].obj["image_path"] == path
