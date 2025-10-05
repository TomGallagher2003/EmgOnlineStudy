from PyQt5 import QtCore

from util.recording import Session


class DeviceInitWorker(QtCore.QThread):
    """Create Session, warm-up/flush, probe capture; emit ready(session) or failed(msg)."""
    ready = QtCore.pyqtSignal(object)   # Session
    failed = QtCore.pyqtSignal(str)

    def __init__(self, use_emg: bool, use_eeg: bool, parent=None):
        super().__init__(parent)
        self.use_emg = use_emg
        self.use_eeg = use_eeg

    def run(self):
        try:
            session = Session(self.use_emg, self.use_eeg)  # sends config on init
            try:
                session.receive_and_ignore(0.75, no_print=True)
                _ = session.get_record(rec_time=0.4, flush=False)
            except Exception as e:
                self.failed.emit(f"An error occurred: {e}")
                return
            self.ready.emit(session)
        except Exception as e:
            self.failed.emit(f"An error occurred: {e}")
