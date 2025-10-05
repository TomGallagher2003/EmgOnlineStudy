from PyQt5 import QtCore

from util.recording import Session


class FlushWorker(QtCore.QThread):
    """
    Repeatedly calls session.receive_and_ignore(chunk_sec, no_print=True)
    to keep the stream fresh while idle. It respects Session.recording
    internally (receive_and_ignore checks it), but we also stop it explicitly
    when a recording begins.
    """
    def __init__(self, session: Session, chunk_sec: float = 0.5, parent=None):
        super().__init__(parent)
        self.session = session
        self.chunk_sec = float(chunk_sec)
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            while not self._stop:
                # This will loop internally for ~chunk_sec seconds, and
                # not read if session.recording is True.
                self.session.receive_and_ignore(self.chunk_sec, no_print=True)
                # yield briefly to the Qt event loop
                self.msleep(10)
        except Exception as e:
            print(f"[flush] error: {e}")
