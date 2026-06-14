"""
SAM 2.1 モデルの ID・設定ファイル名・チェックポイント名を1か所で管理する。

GUI と Worker の双方から参照する純粋なテーブル。torch / sam2 に依存しない。
チェックポイント本体 (*.pt) はリポジトリへコミットしない (.gitignore 済み)。
配置先は既定で models/sam2/ 。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Sam2ModelInfo:
    model_id: str
    display_name: str
    config_name: str        # sam2 の Hydra 設定名 (例: configs/sam2.1/sam2.1_hiera_s.yaml)
    checkpoint_name: str    # models/sam2/ 下のチェックポイントファイル名


# V0.6 初期登録モデル。Large/Tiny は V0.6 では必須ではない。
_MODELS: dict[str, Sam2ModelInfo] = {
    "sam2.1_hiera_small": Sam2ModelInfo(
        model_id="sam2.1_hiera_small",
        display_name="SAM 2.1 Hiera Small",
        config_name="configs/sam2.1/sam2.1_hiera_s.yaml",
        checkpoint_name="sam2.1_hiera_small.pt",
    ),
    "sam2.1_hiera_base_plus": Sam2ModelInfo(
        model_id="sam2.1_hiera_base_plus",
        display_name="SAM 2.1 Hiera Base Plus",
        config_name="configs/sam2.1/sam2.1_hiera_b+.yaml",
        checkpoint_name="sam2.1_hiera_base_plus.pt",
    ),
}

DEFAULT_MODEL_ID = "sam2.1_hiera_small"

# UI に出す順序 (初期選択は先頭)
MODEL_ORDER: tuple[str, ...] = (
    "sam2.1_hiera_small",
    "sam2.1_hiera_base_plus",
)

# 対応する精度モード
PRECISIONS: tuple[str, ...] = ("bf16", "fp16", "fp32")
DEFAULT_PRECISION = "bf16"


def get_model(model_id: str) -> Sam2ModelInfo:
    if model_id not in _MODELS:
        raise KeyError(f"未登録のモデルID: {model_id!r}")
    return _MODELS[model_id]


def has_model(model_id: str) -> bool:
    return model_id in _MODELS


def all_models() -> list[Sam2ModelInfo]:
    return [_MODELS[mid] for mid in MODEL_ORDER]


def checkpoint_path(checkpoint_dir, model_id: str):
    """チェックポイントの絶対パスを Path で返す。"""
    from pathlib import Path
    info = get_model(model_id)
    return Path(checkpoint_dir) / info.checkpoint_name
