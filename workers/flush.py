"""Idle stream flusher for live EMG/EEG sessions.

This module exposes :class:`FlushWorker`, a lightweight ``QThread`` that
periodically drains the device socket via ``Session.receive_and_ignore`` while
the UI is idle. Keeping the stream "fresh" prevents buffer growth and latency
spikes before the next recording.

Behavior:
    - Loops in the background calling
      ``session.receive_and_ignore(chunk_sec, no_print=True)``.
    - Yields briefly to the Qt event loop between chunks (``msleep(10)``).
    - Honors both the internal session check (it won't read during
      ``Session.recording``) and an explicit ``stop()`` signal from the UI.

Typical usage:
    >>> worker = FlushWorker(session, chunk_sec=0.5, parent=self)
    >>> worker.start()
    >>> # before starting a real recording:
    >>> worker.stop()
    >>> worker.wait()  # join if desired
"""

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
        """Initialize the flusher thread.

        Args:
            session: A live, initialized :class:`Session` tied to the device
                socket/transport to be drained while idle.
            chunk_sec: Approximate duration (in seconds) that the underlying
                receive loop should cover per iteration.
            parent: Optional QObject parent for standard Qt ownership semantics.
        """
        super().__init__(parent)
        self.session = session
        self.chunk_sec = float(chunk_sec)
        self._stop = False

    def stop(self) -> None:
        """Request the thread to stop on the next iteration.

        Notes:
            This sets an internal flag checked within :meth:`run`. Call
            ``wait()`` afterward if you need to block until the thread exits.
        """
        self._stop = True

    def run(self) -> None:
        """Drain the device stream in the background until stopped.

        The loop:
            1. Calls ``session.receive_and_ignore(chunk_sec, no_print=True)``.
            2. Sleeps for ~10 ms to yield to the Qt event loop.
            3. Repeats until :meth:`stop` is called or an exception occurs.

        All exceptions are logged to stdout as a best-effort safeguard; the UI
        should remain responsive even if flushing fails.
        """
        try:
            while not self._stop:
                # This will loop internally for ~chunk_sec seconds, and
                # not read if session.recording is True.
                self.session.receive_and_ignore(self.chunk_sec, no_print=True)
                # yield briefly to the Qt event loop
                self.msleep(10)
        except Exception as e:
            print(f"[flush] error: {e}")
