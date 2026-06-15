"""バージョン情報の一元管理。アプリ内で直接バージョン文字列を書かない。"""

APP_NAME = "COLMAP Mask Editor"
APP_VERSION = "0.8"
APP_DISPLAY_NAME = f"{APP_NAME} v{APP_VERSION}"
SETTINGS_SCHEMA_VERSION = 4

# 固定 SAM 2 コミット (sam_backend と整合させる)。AMG manifest へ記録する。
SAM2_COMMIT_SHA = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
