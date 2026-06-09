"""
COLMAP Mask Editor v0.1 - エントリーポイント
"""

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("COLMAP Mask Editor")
    app.setApplicationVersion("0.1")

    window = MainWindow()
    window.show()

    # コマンドライン引数でプロジェクトフォルダを指定可能
    if len(sys.argv) > 1:
        project_path = Path(sys.argv[1])
        if project_path.is_dir():
            window._load_project(project_path)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
