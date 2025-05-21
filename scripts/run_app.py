import sys
from PyQt6.QtWidgets import QApplication
from ephys_recorder import MainWindow


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.recorder.run()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()