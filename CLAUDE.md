# COLMAP Mask Editor — Claude Code 設定

## 開発環境

**conda環境:** `colmap_mask_editor`  
**パス:** `C:\conda-envs\colmap_mask_editor`  
**Python:** 3.12.13  
**conda:** `C:\ProgramData\Anaconda3\Scripts\conda.exe`

## コマンド実行

すべての Python コマンドはこの conda 環境経由で実行する:

```powershell
# Python スクリプト実行
& "C:\ProgramData\Anaconda3\Scripts\conda.exe" run -p "C:\conda-envs\colmap_mask_editor" python <script>

# テスト実行
& "C:\ProgramData\Anaconda3\Scripts\conda.exe" run -p "C:\conda-envs\colmap_mask_editor" python -m pytest colmap_mask_editor/tests/ -v

# パッケージインストール
& "C:\ProgramData\Anaconda3\Scripts\conda.exe" run -p "C:\conda-envs\colmap_mask_editor" pip install <package>
```

## プロジェクト構成

- `colmap_mask_editor/` — メインパッケージ
- `colmap_mask_editor/requirements.txt` — 本番依存関係
- `colmap_mask_editor/requirements-dev.txt` — 開発依存関係（pytest, pytest-qt）

## インストール済みパッケージ（主要）

| パッケージ | バージョン |
|-----------|---------|
| PySide6 | 6.11.1 |
| opencv-python | 4.13.0 |
| numpy | 2.4.6 |
| Pillow | 12.2.0 |
| pytest | 9.1.0 |
| pytest-qt | 4.5.0 |
