"""
COLMAP Mask Editor - エントリーポイント
"""

import logging
import logging.handlers
import sys
from pathlib import Path


def _setup_logging() -> None:
    """アプリケーションログを設定する。RotatingFileHandlerを使用。"""
    log_dir = Path(__file__).parent / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "colmap_mask_editor.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=5_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
    except Exception:
        file_handler = logging.StreamHandler(sys.stderr)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        handlers=[file_handler, console_handler],
    )


def main() -> None:
    _setup_logging()

    from core.version import APP_DISPLAY_NAME, APP_VERSION

    _log = logging.getLogger(__name__)
    _log.info("アプリ起動: %s", APP_DISPLAY_NAME)

    from PySide6.QtCore import QCoreApplication
    from PySide6.QtWidgets import QApplication

    # QSettings用の組織名・アプリ名を QApplication 生成前に設定する
    QCoreApplication.setOrganizationName("KC-nishihana")
    QCoreApplication.setApplicationName("COLMAPMaskEditor")

    app = QApplication(sys.argv)
    app.setApplicationName("COLMAP Mask Editor")
    app.setApplicationVersion(APP_VERSION)

    from ui.main_window import MainWindow

    window = MainWindow()
    window.show()

    if len(sys.argv) > 1:
        project_path = Path(sys.argv[1])
        if project_path.is_dir():
            window._load_project(project_path)

    ret = app.exec()
    _log.info("アプリ終了: %s", APP_DISPLAY_NAME)
    sys.exit(ret)


if __name__ == "__main__":
    main()
