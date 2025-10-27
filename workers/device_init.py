"""Background device/session initialisation worker.

This module provides :class:`DeviceInitWorker`, a ``QThread`` that creates a
:class:`util.recording.Session`, performs a brief warm-up/flush, and probes a
short capture to validate that the device is responsive. It emits either a
``ready(Session)`` signal on success or ``failed(str)`` with a human-readable
error message.

Typical usage:
    >>> worker = DeviceInitWorker(use_emg=True, use_eeg=False, parent=self)
    >>> worker.ready.connect(self.on_session_ready)
    >>> worker.failed.connect(self.on_session_failed)
    >>> worker.start()
"""

from PyQt5 import QtCore

from util.recording import Session


class DeviceInitWorker(QtCore.QThread):
    """Create Session, warm-up/flush, probe capture; emit ready(session) or failed(msg).

    Signals:
        ready (object): Emitted with the initialized :class:`Session` once the
            warm-up and probe capture succeed.
        failed (str): Emitted if session creation, warm-up, or the probe capture
            raises an exception.

    Notes:
        - ``Session`` sends device configuration on initialization; this worker
          purposefully gives the device a short settling period via
          ``receive_and_ignore`` and validates IO with a tiny ``get_record`` call.
        - No UI updates are performed directly; connect to the signals to update
          widgets on the main thread.
    """

    ready = QtCore.pyqtSignal(object)   # Session
    failed = QtCore.pyqtSignal(str)

    def __init__(self, use_emg: bool, use_eeg: bool, parent=None):
        """Initialize the worker.

        Args:
            use_emg: Whether to enable EMG channels in the session.
            use_eeg: Whether to enable EEG channels in the session.
            parent: Optional QObject parent for normal Qt ownership semantics.
        """
        super().__init__(parent)
        self.use_emg = use_emg
        self.use_eeg = use_eeg

    def run(self) -> None:
        """Create and validate a recording session on a background thread.

        Workflow:
            1. Construct :class:`Session` (which applies configuration).
            2. Warm up the stream via ``receive_and_ignore(0.75, no_print=True)``.
            3. Probe a short capture via ``get_record(rec_time=0.4, flush=False)``.
            4. Emit ``ready(session)`` if all steps succeed.
            5. On any exception, emit ``failed(str)`` with details.

        All exceptions are converted to a friendly error string and emitted via
        ``failed``; the thread then exits.
        """
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
