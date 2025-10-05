# movement_timer.py
# PyQt5 app: device select -> parameter entry -> device check/init -> random movement -> arc timer
# -> run_pipeline() -> (classification if EMG: in-label only)
# - Device check/init happens right after parameters screen (session kept open)
# - Recording runs in background; pipeline runs after; classification runs only if EMG present

import sys

from PyQt5 import QtWidgets

from pages.main import MainWindow

# ------------------------------------------------------------------------------

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
