"""
COLMAP Mask Editor v0.4A.1 - エントリーポイント
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

    from PySide6.QtWidgets import QApplication
    from ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("COLMAP Mask Editor")
    app.setApplicationVersion("0.4A.1")

    window = MainWindow()
    window.show()

    if len(sys.argv) > 1:
        project_path = Path(sys.argv[1])
        if project_path.is_dir():
            window._load_project(project_path)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
