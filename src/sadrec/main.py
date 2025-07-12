import sys
from PyQt6.QtWidgets import QApplication
from .gui import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.recorder.run()
    sys.exit(app.exec())
