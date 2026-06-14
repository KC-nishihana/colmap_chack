"""
セットアップ終了コード判定 (純粋関数) のテスト。

PowerShell 本体は実行せず、setup_exit_code() の判定ロジックのみ検証する。
setup_sam2_cuda_windows.ps1 はこの関数と同じ規則で終了コードを返す。
"""

from ai.cuda_verification import setup_exit_code


def test_no_checkpoint_without_buildonly_is_4():
    # チェックポイントなし・BuildOnly なし -> 4 (実機検証未完了)
    assert setup_exit_code(extension_imported=True, checkpoint_present=False,
                           build_only=False) == 4


def test_no_checkpoint_with_buildonly_is_0():
    # チェックポイントなし・BuildOnly あり -> 0 (ビルド+import のみ成功)
    assert setup_exit_code(extension_imported=True, checkpoint_present=False,
                           build_only=True) == 0


def test_extension_import_failed_is_3():
    assert setup_exit_code(extension_imported=False, checkpoint_present=True,
                           build_only=False) == 3


def test_extension_import_failed_even_buildonly_is_3():
    assert setup_exit_code(extension_imported=False, checkpoint_present=False,
                           build_only=True) == 3


def test_kernel_execution_failed_is_3():
    assert setup_exit_code(extension_imported=True, checkpoint_present=True,
                           build_only=False, kernel_executed=False,
                           full_verification_ok=False) == 3


def test_full_verification_failed_is_3():
    assert setup_exit_code(extension_imported=True, checkpoint_present=True,
                           build_only=False, kernel_executed=True,
                           full_verification_ok=False) == 3


def test_full_success_is_0():
    assert setup_exit_code(extension_imported=True, checkpoint_present=True,
                           build_only=False, kernel_executed=True,
                           full_verification_ok=True) == 0
