from PyQt5 import QtCore

from util.recording import Session


class RecordingWorker(QtCore.QThread):
    finished_ok = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)
    capture_started = QtCore.pyqtSignal()

    def __init__(self, session: Session, rec_len: float, parent=None):
        super().__init__(parent)
        self.session = session
        self.rec_len = rec_len

    def run(self):
        try:
            self.capture_started.emit()
            data = self.session.get_record(rec_time=self.rec_len, flush=False)
            if data is None or data.size == 0:
                self.failed.emit("Recording returned no data.")
                return
            self.finished_ok.emit(data)
        except Exception as e:
            self.failed.emit(str(e))

