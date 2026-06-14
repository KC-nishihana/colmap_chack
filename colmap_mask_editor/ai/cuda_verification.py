"""
CUDA 拡張検証・セットアップの「成功判定ロジック」を純粋関数として分離する。

このモジュールは torch / sam2 / sam2._C を一切 import しない純粋ロジックであり、
通常 pytest からテスト可能。verify スクリプト (実機) と setup スクリプトの判定が
ここに集約され、テストと実装が同じ基準を共有する。

V0.6.1 方針:
  - sam2._C の import 成功「だけ」を成功扱いにしない。
  - get_connected_components の CUDA カーネル実行・fill_holes 後処理・
    実モデル推論まで成功した場合のみ成功とする。
  - チェックポイントが無い実機未検証を「完全成功」にしない。
"""

from __future__ import annotations

# verify スクリプトの最終成功に必須となるチェック項目 (result dict のキー)。
REQUIRED_VERIFICATION_KEYS: tuple[str, ...] = (
    "cuda_extension_imported",
    "cuda_extension_kernel_executed",
    "fill_holes_test",
    "model_loaded",
    "embedding_ok",
    "positive_click_ok",
    "negative_click_ok",
    "box_prompt_ok",
    "multimask_output",
)


def evaluate_verification(result: dict) -> bool:
    """
    verify_sam2_cuda_extension.py の検証結果 dict から最終成功を判定する。

    すべての必須項目が厳密に True の場合のみ True。
    以下「だけ」では成功にしない:
      - sam2._C を import できた
      - SAM 2 モデルをロードできた
      - NumPy マスクが 0/255 だった
      - 推論結果が 1 件返った
    """
    return all(result.get(key) is True for key in REQUIRED_VERIFICATION_KEYS)


# セットアップ終了コードの意味 (README / setup ps1 と統一):
#   0 = 完全検証成功 (または BuildOnly でのビルド+import 成功)
#   1 = 環境または入力不足
#   2 = PyTorch CUDA と CUDA Toolkit のバージョン不整合
#   3 = SAM 2 / CUDA 拡張のビルド・ロード・実行失敗
#   4 = チェックポイント不足により実機検証未完了
EXIT_SUCCESS = 0
EXIT_ENV_INSUFFICIENT = 1
EXIT_VERSION_MISMATCH = 2
EXIT_EXTENSION_FAILED = 3
EXIT_CHECKPOINT_MISSING = 4


def setup_exit_code(
    *,
    extension_imported: bool,
    checkpoint_present: bool,
    build_only: bool = False,
    kernel_executed: bool = False,
    full_verification_ok: bool = False,
) -> int:
    """
    CUDA 拡張ビルド後のセットアップ終了コードを決定する純粋関数。

    判定順:
      1. 拡張 import 失敗            -> 3
      2. BuildOnly (import まで成功)  -> 0 (実機検証は行わない)
      3. チェックポイント無し         -> 4 (実機検証未完了)
      4. CUDA カーネル未実行          -> 3
      5. 実機フル検証 NG              -> 3
      6. すべて成功                   -> 0

    注: 環境不足 (1) や CUDA バージョン不整合 (2) は拡張ビルド以前の段階で
    判定されるため、この関数は「ビルド後」の 0/3/4 を扱う。
    """
    if not extension_imported:
        return EXIT_EXTENSION_FAILED
    if build_only:
        return EXIT_SUCCESS
    if not checkpoint_present:
        return EXIT_CHECKPOINT_MISSING
    if not kernel_executed:
        return EXIT_EXTENSION_FAILED
    if not full_verification_ok:
        return EXIT_EXTENSION_FAILED
    return EXIT_SUCCESS
