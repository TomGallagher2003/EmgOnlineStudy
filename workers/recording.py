# workers/recording.py
from typing import Optional, List
import numpy as np
from PyQt5 import QtCore
from util.recording import Session


class RecordingWorker(QtCore.QThread):
    finished_ok = QtCore.pyqtSignal(object)    # (channels, total_samples)
    failed = QtCore.pyqtSignal(str)
    capture_started = QtCore.pyqtSignal()
    sofar_ready = QtCore.pyqtSignal(object)    # (channels, samples_so_far), cumulative

    def __init__(
        self,
        session: Session,
        rec_len: float,
        parent: Optional[QtCore.QObject] = None,
        *,
        chunk_sec: float = 0.05,   # 50 ms @ 2 kHz ≈ 100 samples
        emit_every_n_chunks: int = 1,  # throttle UI updates if you like
    ):
        super().__init__(parent)
        self.session = session
        self.rec_len = float(max(0.0, rec_len))
        self.chunk_sec = float(max(0.005, chunk_sec))
        self.emit_every_n_chunks = max(1, int(emit_every_n_chunks))

    def run(self) -> None:
        try:
            if self.isInterruptionRequested():
                self.failed.emit("Recording interrupted before start.")
                return
            if self.rec_len <= 0.0:
                self.failed.emit("Recording length must be > 0.")
                return

            self.capture_started.emit()

            # One-shot path if chunk >= total
            if self.chunk_sec >= self.rec_len:
                data = self.session.get_record(rec_time=self.rec_len, flush=False)
                if data is None or getattr(data, "size", 0) == 0:
                    self.failed.emit("Recording returned no data.")
                    return
                a = np.asarray(data)
                if a.ndim == 1:
                    a = a.reshape(1, -1)
                elif a.ndim == 2 and a.shape[0] > a.shape[1]:
                    # if somehow (samples, channels), transpose to (channels, samples)
                    a = a.T
                self.sofar_ready.emit(a)   # final snapshot
                self.finished_ok.emit(a)
                return

            # Chunked path
            remaining = self.rec_len
            chunks: List[np.ndarray] = []
            nchunks = 0

            while remaining > 1e-6 and not self.isInterruptionRequested():
                step = min(self.chunk_sec, remaining)
                try:
                    chunk = self.session.get_record(rec_time=step, flush=False)
                except Exception:
                    chunk = None

                if chunk is not None and getattr(chunk, "size", 0) > 0:
                    a = np.asarray(chunk)
                    if a.ndim == 1:
                        a = a.reshape(1, -1)
                    elif a.ndim == 2 and a.shape[0] > a.shape[1]:
                        # ensure (channels, samples)
                        a = a.T
                    chunks.append(a)
                    nchunks += 1

                    if (nchunks % self.emit_every_n_chunks) == 0:
                        # Build cumulative snapshot for the UI
                        sofar = np.concatenate(chunks, axis=1)
                        self.sofar_ready.emit(sofar)

                remaining -= step

            if not chunks:
                self.failed.emit("Recording returned no data.")
                return

            full = np.concatenate(chunks, axis=1)  # (channels, total_samples)
            # Ensure final snapshot emitted at least once
            self.sofar_ready.emit(full)
            self.finished_ok.emit(full)

        except Exception as e:
            self.failed.emit(str(e))
