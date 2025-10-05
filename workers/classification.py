from typing import Optional

import numpy as np
from PyQt5 import QtCore

from pipeline_sections.classification import classify_emg


class ClassificationWorker(QtCore.QThread):
    """Runs post-pipeline classification on the IN-LABEL subset only."""
    finished_ok = QtCore.pyqtSignal(object)  # emits (A, B)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, processed_emg_inlabel: Optional[np.ndarray], parent=None):
        super().__init__(parent)
        self.processed_emg_inlabel = processed_emg_inlabel

    def run(self):
        try:
            arr = self.processed_emg_inlabel
            if arr is None or getattr(arr, "size", 0) == 0 or getattr(arr, "shape", (0,))[0] == 0:
                self.finished_ok.emit(("", 0.0))
                return
            result = classify_emg(arr)
            self.finished_ok.emit(result)
        except Exception as e:
            self.failed.emit(str(e))

