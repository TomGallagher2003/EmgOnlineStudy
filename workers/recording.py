from PyQt5 import QtCore

from util.recording import Session

"""Background recording worker thread.

This module provides :class:`RecordingWorker`, a thin QThread wrapper that
records for a fixed duration using an existing :class:`util.recording.Session`
and emits signals for UI coordination. It is designed for use in PyQt5 apps
where the main thread must remain responsive while EMG/EEG capture runs.

Example:
    >>> # inside a QWidget or MainWindow
    >>> worker = RecordingWorker(session, rec_len=5.0, parent=self)
    >>> worker.capture_started.connect(lambda: print("Capture started"))
    >>> worker.finished_ok.connect(lambda data: print("Got", data.shape))
    >>> worker.failed.connect(lambda msg: print("Failed:", msg))
    >>> worker.start()
"""


class RecordingWorker(QtCore.QThread):
    """Run a fixed-length recording in a background thread.

    Signals:
        finished_ok (object): Emitted with the recorded data (e.g., NumPy array)
            when capture succeeds.
        failed (str): Emitted with an error message if capture fails or returns
            no data.
        capture_started (): Emitted immediately before recording begins to allow
            the UI to update state (e.g., disable buttons or show spinners).

    Notes:
        - This worker **does not** create or configure the device; it assumes the
          provided :class:`util.recording.Session` is already initialized and ready.
        - The thread emits exactly one terminal signal: either ``finished_ok`` or
          ``failed``. Consumers should disconnect or delete the worker afterward.
    """

    finished_ok = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)
    capture_started = QtCore.pyqtSignal()

    def __init__(self, session: Session, rec_len: float, parent=None):
        """Initialize the worker.

        Args:
            session: A pre-initialized recording session used to acquire data.
            rec_len: Desired recording duration in seconds.
            parent: Optional QObject parent for normal Qt ownership semantics.

        """
        super().__init__(parent)
        self.session = session
        self.rec_len = rec_len

    def run(self) -> None:
        """Execute the capture on a background thread.

        Workflow:
            1. Emit ``capture_started`` to allow the UI to react.
            2. Call ``session.get_record(rec_time=rec_len, flush=False)``.
            3. If no data is returned, emit ``failed``.
            4. Otherwise, emit ``finished_ok`` with the data.

        Emitted:
            capture_started: Before starting the blocking capture call.
            finished_ok: On successful capture with the data payload.
            failed: On exceptions or empty results, with a human-readable message.
        """
        try:
            self.capture_started.emit()
            data = self.session.get_record(rec_time=self.rec_len, flush=False)
            if data is None or getattr(data, "size", 0) == 0:
                self.failed.emit("Recording returned no data.")
                return
            self.finished_ok.emit(data)
        except Exception as e:
            self.failed.emit(str(e))
