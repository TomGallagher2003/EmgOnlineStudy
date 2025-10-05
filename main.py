# movement_timer.py
# PyQt5 app: device select -> parameter entry -> device check/init -> random movement -> arc timer
# -> run_pipeline() -> (classification if EMG: in-label only)
# - Device check/init happens right after parameters screen (session kept open)
# - Recording runs in background; pipeline runs after; classification runs only if EMG present

import os
import sys
import time
import random
from typing import Optional, Dict, Any, Tuple

import h5py
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from pipeline_sections.classification import classify_emg
from pipeline_sections.filters import selective_filter
from pipeline_sections.reduce_eeg_samples import reduce_eeg
from util.images import Images
from recording import Session  # adjust import path if needed
from util.movement_segmentation import detect_movement_mask
from pipeline_sections.normalisation import normalise_data
from pipeline_sections.windows import window_data

# ------------------------------------------------------------------------------

# Expecting util.images to expose MOVEMENT_TUPLES = list[(clean_name, filename)]
MOVEMENTS = Images.MOVEMENT_TUPLES

# Faster than CSV if you want speedier saves inside run_pipeline
SAVE_AS_NPY = False

DEFAULT_WINDOW_MS = 256.0  # default window size (ms)


class ArcTimerWidget(QtWidgets.QWidget):
    finished = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._progress = 0.0
        self._duration_ms = 0          # default to 0 so we don't show 4.0s
        self._tick_ms = 30
        self._elapsed = 0
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self.setMinimumSize(160, 160)

    def set_duration(self, duration_ms: int):
        """Prime the timer display without starting it."""
        self._duration_ms = max(1, int(duration_ms))
        self._elapsed = 0
        self._progress = 0.0
        self.update()

    def start(self, duration_ms=4000):
        # keep start() flexible, but we’ll usually call set_duration() beforehand
        self._duration_ms = max(1, int(duration_ms))
        self._elapsed = 0
        self._progress = 0.0
        self._timer.start(self._tick_ms)
        self.update()

    def stop(self):
        if self._timer.isActive():
            self._timer.stop()
        self._progress, self._elapsed = 0.0, 0
        self.update()

    def is_running(self):
        return self._timer.isActive()

    def _on_tick(self):
        self._elapsed += self._tick_ms
        self._progress = min(1.0, self._elapsed / self._duration_ms)
        self.update()
        if self._progress >= 1.0:
            self._timer.stop()
            self.finished.emit()

    def paintEvent(self, event):
        side = min(self.width(), self.height())
        rect = QtCore.QRect(
            (self.width() - side) // 2,
            (self.height() - side) // 2,
            side,
            side,
        )
        start_angle = 90 * 16
        span_angle = -int(self._progress * 360 * 16)

        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)

        # background circle
        bg_pen = QtGui.QPen(QtGui.QColor(220, 220, 220), 12)
        p.setPen(bg_pen)
        p.drawEllipse(rect.adjusted(10, 10, -10, -10))

        # foreground arc
        fg_pen = QtGui.QPen(QtGui.QColor(70, 120, 255), 12, cap=QtCore.Qt.RoundCap)
        p.setPen(fg_pen)
        p.drawArc(rect.adjusted(10, 10, -10, -10), start_angle, span_angle)

        # countdown text
        remaining_ms = max(0, self._duration_ms - self._elapsed)
        secs = remaining_ms / 1000.0
        p.setPen(QtGui.QColor(50, 50, 50))
        font = p.font()
        font.setPointSize(int(side * 0.12))
        p.setFont(font)
        p.drawText(rect, QtCore.Qt.AlignCenter, f"{secs:0.1f}s")


class DeviceSelectPage(QtWidgets.QWidget):
    proceed = QtCore.pyqtSignal(bool, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        title = QtWidgets.QLabel("Select Devices")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: 600;")

        self.cb_emg = QtWidgets.QCheckBox("Use EMG (Muovi)")
        self.cb_emg.setChecked(True)
        self.cb_eeg = QtWidgets.QCheckBox("Use EEG (Muovi+)")
        self.cb_eeg.setChecked(True)

        btn_continue = QtWidgets.QPushButton("Continue")
        btn_continue.setFixedHeight(36)
        btn_continue.clicked.connect(self._on_continue)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addSpacing(16)
        layout.addWidget(title)
        layout.addSpacing(12)
        layout.addWidget(self.cb_emg)
        layout.addWidget(self.cb_eeg)
        layout.addStretch(1)
        layout.addWidget(btn_continue)

    def _on_continue(self):
        self.proceed.emit(self.cb_emg.isChecked(), self.cb_eeg.isChecked())


class ParametersPage(QtWidgets.QWidget):
    """Collect trial, recording length, EMG/EEG filters, and window settings (default 256ms; optional custom)."""

    proceed = QtCore.pyqtSignal(dict)
    FILTER_OPTIONS = ["None", "Pass", "Notch"]

    class FilterRow(QtWidgets.QWidget):
        changed = QtCore.pyqtSignal()
        def __init__(self, label_text: str, parent=None):
            super().__init__(parent)
            self.type_box = QtWidgets.QComboBox()
            self.type_box.addItems(ParametersPage.FILTER_OPTIONS)

            # Pass row (Lower/Upper)
            self.pass_row = QtWidgets.QWidget()
            pr_layout = QtWidgets.QHBoxLayout(self.pass_row); pr_layout.setContentsMargins(0,0,0,0)
            self.lower_edit = QtWidgets.QLineEdit(); self.upper_edit = QtWidgets.QLineEdit()
            self.lower_edit.setPlaceholderText("Lower Hz (optional)")
            self.upper_edit.setPlaceholderText("Upper Hz (optional)")
            self.lower_edit.setValidator(QtGui.QDoubleValidator(0.0, 1e6, 3, self))
            self.upper_edit.setValidator(QtGui.QDoubleValidator(0.0, 1e6, 3, self))
            pr_layout.addWidget(QtWidgets.QLabel("Lower:")); pr_layout.addWidget(self.lower_edit)
            pr_layout.addSpacing(8)
            pr_layout.addWidget(QtWidgets.QLabel("Upper:")); pr_layout.addWidget(self.upper_edit)

            # Notch row (Center)
            self.notch_row = QtWidgets.QWidget()
            nr_layout = QtWidgets.QHBoxLayout(self.notch_row); nr_layout.setContentsMargins(0,0,0,0)
            self.center_edit = QtWidgets.QLineEdit()
            self.center_edit.setPlaceholderText("Center Hz")
            self.center_edit.setValidator(QtGui.QDoubleValidator(0.0, 1e6, 3, self))
            nr_layout.addWidget(QtWidgets.QLabel("Center:")); nr_layout.addWidget(self.center_edit)

            # None row
            self.none_row = QtWidgets.QWidget()
            nr = QtWidgets.QHBoxLayout(self.none_row); nr.setContentsMargins(0,0,0,0)
            nr.addWidget(QtWidgets.QLabel("")); nr.addStretch(1)

            # Layout
            left = QtWidgets.QVBoxLayout(); left.setContentsMargins(0,0,0,0)
            left.addWidget(QtWidgets.QLabel(label_text)); left.addWidget(self.type_box)
            right = QtWidgets.QVBoxLayout(); right.setContentsMargins(0,0,0,0)
            right.addWidget(self.none_row); right.addWidget(self.pass_row); right.addWidget(self.notch_row)
            lay = QtWidgets.QHBoxLayout(self); lay.addLayout(left); lay.addSpacing(10); lay.addLayout(right, 1)

            self.type_box.currentTextChanged.connect(self._update_rows)
            for w in (self.lower_edit, self.upper_edit, self.center_edit):
                w.textChanged.connect(self.changed.emit)
            self._update_rows()

        def _update_rows(self):
            t = self.type_box.currentText()
            self.none_row.setVisible(t == "None")
            self.pass_row.setVisible(t == "Pass")
            self.notch_row.setVisible(t == "Notch")
            self.changed.emit()

        def value(self) -> dict:
            def f(le: QtWidgets.QLineEdit):
                txt = le.text().strip()
                return float(txt) if txt else None
            t = self.type_box.currentText()
            out = {"type": t, "lower": None, "upper": None, "center": None}
            if t == "Pass":
                out["lower"], out["upper"] = f(self.lower_edit), f(self.upper_edit)
            elif t == "Notch":
                out["center"] = f(self.center_edit)
            return out

        def validate(self, parent: QtWidgets.QWidget) -> bool:
            t = self.type_box.currentText()
            if t == "Pass":
                lower = self.lower_edit.text().strip()
                upper = self.upper_edit.text().strip()
                if not lower and not upper:
                    QtWidgets.QMessageBox.warning(parent, "Invalid Pass filter",
                        "Enter at least one of Lower Hz or Upper Hz for a Pass filter.")
                    return False
                if lower and upper:
                    if float(lower) >= float(upper):
                        QtWidgets.QMessageBox.warning(parent, "Invalid Pass band",
                            "Lower Hz must be strictly less than Upper Hz.")
                        return False
            elif t == "Notch":
                if not self.center_edit.text().strip():
                    QtWidgets.QMessageBox.warning(parent, "Invalid Notch filter",
                        "Center Hz is required for a Notch filter.")
                    return False
            return True

    def __init__(self, use_emg: bool, use_eeg: bool, parent=None):
        super().__init__(parent)
        self.use_emg = use_emg
        self.use_eeg = use_eeg
        self._custom_window_enabled = False

        title = QtWidgets.QLabel("Experiment Parameters")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: 600;")

        # Trial / Length
        self.trial_edit = QtWidgets.QLineEdit(); self.trial_edit.setPlaceholderText("")
        self.length_edit = QtWidgets.QLineEdit(); self.length_edit.setPlaceholderText("")
        self.length_edit.setValidator(QtGui.QDoubleValidator(0.001, 1e6, 3, self))

        base_form = QtWidgets.QFormLayout()
        base_form.addRow("Trial number:", self.trial_edit)
        base_form.addRow("Recording length (s):", self.length_edit)

        # Window size: default 256ms (hidden input), with "Custom…" to reveal input + warning
        self.window_default_label = QtWidgets.QLabel(f"Window size: {int(DEFAULT_WINDOW_MS)} ms (default)")
        self.window_custom_btn = QtWidgets.QPushButton("Custom…")
        self.window_custom_btn.setFixedWidth(100)
        self.window_custom_btn.clicked.connect(self._toggle_custom_window)

        window_row = QtWidgets.QHBoxLayout()
        window_row.addWidget(self.window_default_label)
        window_row.addSpacing(12)
        window_row.addWidget(self.window_custom_btn)
        window_row.addStretch(1)

        # Hidden custom row
        self.custom_row = QtWidgets.QWidget()
        cr_layout = QtWidgets.QVBoxLayout(self.custom_row); cr_layout.setContentsMargins(0,0,0,0)
        self.window_ms_edit = QtWidgets.QLineEdit()
        self.window_ms_edit.setValidator(QtGui.QDoubleValidator(0.001, 1e9, 3, self))
        self.window_ms_edit.setPlaceholderText("e.g., 256")
        warn = QtWidgets.QLabel("Warning: mismatched window sizes may cause classification errors.")
        warn.setStyleSheet("color:#b00; font-size: 11px;")
        cr_layout.addWidget(QtWidgets.QLabel("Custom window size (ms):"))
        cr_layout.addWidget(self.window_ms_edit)
        cr_layout.addWidget(warn)
        self.custom_row.setVisible(False)

        # Overlap (ms) still available (visible)
        self.overlap_ms_edit = QtWidgets.QLineEdit()
        self.overlap_ms_edit.setValidator(QtGui.QDoubleValidator(0.0, 1e9, 3, self))
        self.overlap_ms_edit.setPlaceholderText("e.g., 128")
        base_form.addRow(window_row)
        base_form.addRow(self.custom_row)
        base_form.addRow("Overlap (ms):", self.overlap_ms_edit)

        # Normalisation toggle
        self.use_normalisation = QtWidgets.QCheckBox("Normalise Data")
        self.use_normalisation.setChecked(True)
        base_form.addWidget(self.use_normalisation)

        # Auto segmentation toggle (EMG only)
        self.emg_auto_seg = QtWidgets.QCheckBox("Use automatic movement segmentation (EMG)")
        self.emg_auto_seg.setChecked(False)
        self.emg_auto_unavailable_label = QtWidgets.QLabel("Automatic segmentation is unavailable for EEG-only trials")
        self.emg_auto_unavailable_label.setVisible(not self.use_emg)
        base_form.addWidget(self.emg_auto_seg)
        base_form.addWidget(self.emg_auto_unavailable_label)

        # EMG filters group (only visible if EMG selected)
        self.emg_group = QtWidgets.QGroupBox("EMG Filters")
        emg_lay = QtWidgets.QFormLayout(self.emg_group)
        self.emg_a = ParametersPage.FilterRow("First Filter")
        self.emg_b = ParametersPage.FilterRow("Second Filter")
        self.emg_c = ParametersPage.FilterRow("Third Filter")
        emg_lay.addRow(self.emg_a); emg_lay.addRow(self.emg_b); emg_lay.addRow(self.emg_c)
        self.emg_group.setVisible(self.use_emg)

        # EEG filters group (only visible if EEG selected)
        self.eeg_group = QtWidgets.QGroupBox("EEG Filters")
        eeg_lay = QtWidgets.QFormLayout(self.eeg_group)
        self.eeg_a = ParametersPage.FilterRow("First Filter")
        self.eeg_b = ParametersPage.FilterRow("Second Filter")
        self.eeg_c = ParametersPage.FilterRow("Third Filter")
        eeg_lay.addRow(self.eeg_a); eeg_lay.addRow(self.eeg_b); eeg_lay.addRow(self.eeg_c)
        self.eeg_group.setVisible(self.use_eeg)

        # Buttons
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_back = QtWidgets.QPushButton("Back")
        self.btn_next = QtWidgets.QPushButton("Continue"); self.btn_next.setDefault(True)
        btn_row.addStretch(1); btn_row.addWidget(self.btn_back); btn_row.addWidget(self.btn_next)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addSpacing(12); layout.addWidget(title); layout.addSpacing(8)
        layout.addLayout(base_form)
        layout.addWidget(self.emg_group)
        layout.addWidget(self.eeg_group)
        layout.addStretch(1)
        layout.addLayout(btn_row)

        self.btn_next.clicked.connect(self._on_continue)

    def _toggle_custom_window(self):
        self._custom_window_enabled = not self._custom_window_enabled
        self.custom_row.setVisible(self._custom_window_enabled)
        self.window_custom_btn.setText("Default" if self._custom_window_enabled else "Custom…")
        if not self._custom_window_enabled:
            # Clear custom input when returning to default
            self.window_ms_edit.clear()

    def _validate_filter_group(self, rows) -> bool:
        for r in rows:
            if not r.validate(self):
                return False
        return True

    def _on_continue(self):
        # Trial
        t = self.trial_edit.text().strip()
        try:
            trial_num = int(t)
        except Exception:
            QtWidgets.QMessageBox.warning(self, "Invalid trial number", "Trial number must be an integer.")
            return

        # Length
        lt = self.length_edit.text().strip()
        try:
            rec_len = float(lt)
        except Exception:
            QtWidgets.QMessageBox.warning(self, "Invalid recording length", "Recording length must be a number (seconds).")
            return
        if rec_len <= 0:
            QtWidgets.QMessageBox.warning(self, "Invalid recording length", "Recording length must be > 0.")
            return

        # Window size (ms): default or custom
        if self._custom_window_enabled:
            w_txt = self.window_ms_edit.text().strip()
            if not w_txt:
                QtWidgets.QMessageBox.warning(self, "Missing window size", "Enter a custom window size (ms), or click Default.")
                return
            try:
                window_ms = float(w_txt)
            except Exception:
                QtWidgets.QMessageBox.warning(self, "Invalid window size", "Window size (ms) must be a number.")
                return
            if window_ms <= 0:
                QtWidgets.QMessageBox.warning(self, "Invalid window size", "Window size (ms) must be > 0.")
                return
        else:
            window_ms = DEFAULT_WINDOW_MS

        # Overlap size (ms)
        o_txt = self.overlap_ms_edit.text().strip()
        try:
            overlap_ms = float(o_txt) if o_txt else 0.0
        except Exception:
            QtWidgets.QMessageBox.warning(self, "Invalid overlap", "Overlap (ms) must be a number.")
            return
        if overlap_ms < 0:
            QtWidgets.QMessageBox.warning(self, "Invalid overlap", "Overlap (ms) must be ≥ 0.")
            return
        if overlap_ms >= window_ms:
            QtWidgets.QMessageBox.warning(self, "Invalid overlap", "Overlap (ms) must be less than Window size (ms).")
            return

        # Validate visible groups
        if self.use_emg and not self._validate_filter_group((self.emg_a, self.emg_b, self.emg_c)):
            return
        if self.use_eeg and not self._validate_filter_group((self.eeg_a, self.eeg_b, self.eeg_c)):
            return

        filters_struct = {
            "emg": [self.emg_a.value(), self.emg_b.value(), self.emg_c.value()] if self.use_emg else [],
            "eeg": [self.eeg_a.value(), self.eeg_b.value(), self.eeg_c.value()] if self.use_eeg else [],
        }

        params = {
            "trial": trial_num,
            "recording_length": rec_len,
            "window_ms": window_ms,
            "overlap_ms": overlap_ms,
            "filters": filters_struct,
            "use_auto": self.emg_auto_seg.isChecked(),
            "use_normalisation": self.use_normalisation.isChecked()
        }
        self.proceed.emit(params)


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


# NEW: continuous idle flusher until a recording starts
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


class PipelineWorker(QtCore.QThread):
    """Runs the data processing pipeline and returns (processed_emg_all, processed_emg_inlabel)."""
    finished_ok = QtCore.pyqtSignal(object)  # emits (processed_emg_all, processed_emg_inlabel)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, pipeline_fn, movement_name, data, parent=None):
        super().__init__(parent)
        self.pipeline_fn = pipeline_fn
        self.movement_name = movement_name
        self.data = data

    def run(self):
        try:
            processed_tuple = self.pipeline_fn(self.movement_name, self.data)
            self.finished_ok.emit(processed_tuple)
        except Exception as e:
            self.failed.emit(str(e))


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


class ExperimentPage(QtWidgets.QWidget):
    """Random movement + Start — session is initialized on entry (device check happens here)."""

    def __init__(self, use_emg, use_eeg, params: Optional[Dict[str, Any]] = None, parent=None):
        super().__init__(parent)
        self.use_emg = use_emg
        self.use_eeg = use_eeg
        self.params = params or {}
        self.current_movement = None           # tuple (name, filename)
        self.current_movement_idx = None       # int index into MOVEMENTS
        self._last_movement_idx = None
        self.session: Optional[Session] = None
        self.recording_worker = None
        self._pipeline_worker = None
        self._classification_worker = None
        self.is_pipeline_running = False
        self.is_classifying = False
        self.recording_done = False  # track when recording thread is actually done
        self._current_pixmap_path = None
        self._flusher: Optional[FlushWorker] = None

        # UI - make image large
        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setMinimumHeight(420)
        self.image_label.setStyleSheet("background: #fafafa; border: 1px solid #e6e6e6;")
        self.image_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self.status_label = QtWidgets.QLabel("Initializing devices…")
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.status_label.setStyleSheet("color:#666;")

        self.btn_random = QtWidgets.QPushButton("Get Random Movement")
        self.btn_random.clicked.connect(self.pick_random_movement)
        self.btn_random.setEnabled(False)  # disabled until devices ready

        self.btn_start = QtWidgets.QPushButton("Start Recording")
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self.start_recording)

        self.arc = ArcTimerWidget()
        self.arc.finished.connect(self._on_arc_complete)

        if "recording_length" in self.params:
            self.arc.set_duration(int(self.params["recording_length"] * 1000))

        # --- Classification Results panel (EMG only, single line) ---
        self.results_box = QtWidgets.QGroupBox("Classification Results")
        rb_layout = QtWidgets.QVBoxLayout(self.results_box)
        self.results_label = QtWidgets.QLabel("Not yet classified")
        self.results_label.setWordWrap(True)
        self.results_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.results_label.setMinimumHeight(24)
        rb_layout.addWidget(self.results_label)
        self.results_box.setVisible(self.use_emg)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.image_label, 3)
        layout.addSpacing(8)
        layout.addWidget(self.status_label)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.btn_random)
        row.addWidget(self.btn_start)
        layout.addLayout(row)
        layout.addWidget(self.arc, 1)
        layout.addWidget(self.results_box)  # hidden if not EMG
        layout.addStretch(1)

        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        # Kick off device init immediately
        self._init_worker = DeviceInitWorker(self.use_emg, self.use_eeg, parent=self)
        self._init_worker.ready.connect(self._on_devices_ready)
        self._init_worker.failed.connect(self._on_devices_failed)
        self._init_worker.start()

    # scale image on resize
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._current_pixmap_path and os.path.exists(self._current_pixmap_path):
            pix = QtGui.QPixmap(self._current_pixmap_path)
            if not pix.isNull():
                scaled = pix.scaled(
                    self.image_label.size() * 0.98,
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation,
                )
                self.image_label.setPixmap(scaled)

    # ----- Flusher helpers -----
    def _start_flusher(self):
        if self.session is None:
            return
        if self._flusher is not None and self._flusher.isRunning():
            return
        self._flusher = FlushWorker(self.session, chunk_sec=0.5, parent=self)
        self._flusher.start()
        print("[flush] started")

    def _stop_flusher(self):
        if self._flusher is not None:
            self._flusher.stop()
            # wait briefly to ensure it stops before recording starts
            self._flusher.wait(500)
            print("[flush] stopped")

    # ----- Device init callbacks -----
    def _on_devices_ready(self, session: Session):
        self.session = session
        self.status_label.setText("Devices ready.")
        self.btn_random.setEnabled(True)
        self.btn_start.setEnabled(self.current_movement is not None)
        # start background flushing as soon as session is open
        self._start_flusher()

    def _on_devices_failed(self, msg: str):
        self.status_label.setText(msg)

    # ----- Experiment flow -----
    def pick_random_movement(self):
        # Better randomness + avoid immediate repeats
        rng = random.SystemRandom()
        if len(MOVEMENTS) > 1:
            while True:
                new_idx = rng.randrange(len(MOVEMENTS))
                if new_idx != self._last_movement_idx:
                    break
        else:
            new_idx = 0
        self._last_movement_idx = new_idx
        self.current_movement_idx = new_idx
        self.current_movement = MOVEMENTS[self.current_movement_idx]

        # expose movement_id for pipeline/classification use; 1..N
        self.params["movement_id"] = int(self.current_movement_idx) + 1

        # load image
        _, filename = self.current_movement
        img_path = os.path.join(self.script_dir, filename)
        self._current_pixmap_path = img_path if os.path.exists(img_path) else None
        if self._current_pixmap_path:
            pix = QtGui.QPixmap(self._current_pixmap_path)
            if not pix.isNull():
                scaled = pix.scaled(
                    self.image_label.size() * 0.98,
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation,
                )
                self.image_label.setPixmap(scaled)
            else:
                self.image_label.clear()
                self._current_pixmap_path = None
        else:
            self.image_label.clear()

        # Enable start if devices ready
        if self.session is not None:
            self.btn_start.setEnabled(True)

    def start_recording(self):
        if self.arc.is_running() or self.session is None:
            return
        if self.current_movement is None:
            QtWidgets.QMessageBox.information(self, "Pick a movement", "Please choose a movement first.")
            return

        # Stop the idle flusher right before starting to record
        self._stop_flusher()

        self.status_label.setText("Recording…")
        self.btn_random.setEnabled(False)
        self.btn_start.setEnabled(False)
        self.recording_done = False
        # Reset results panel for the new attempt (only if EMG path is visible)
        if self.use_emg:
            self.results_label.setText("Classification Results: placeholder")

        self.recording_worker = RecordingWorker(self.session, self.params["recording_length"], parent=self)
        # start arc on actual capture start; duration follows user-set recording length
        self.recording_worker.capture_started.connect(lambda: self.arc.start(int(self.params["recording_length"] * 1000)))
        self.recording_worker.finished_ok.connect(self._on_recording_finished)
        self.recording_worker.failed.connect(self._on_recording_failed)
        self.recording_worker.start()

    def _on_recording_failed(self, msg: str):
        self.status_label.setText(f"Error: {msg}")
        self.btn_random.setEnabled(True)
        self.btn_start.setEnabled(True)
        self.arc.stop()
        self.recording_done = True
        # Resume flushing since we're idle again
        self._start_flusher()

    def _on_recording_finished(self, data):
        self.recording_done = True
        # Immediately resume flushing while we process, to keep stream fresh
        self._start_flusher()

        movement_name = self.current_movement[0] if self.current_movement else None

        # Start PIPELINE step
        self.is_pipeline_running = True
        self.btn_random.setEnabled(False)
        self.btn_start.setEnabled(False)
        self.status_label.setText("Processing data...")

        self._pipeline_worker = PipelineWorker(self.run_pipeline, movement_name, data, parent=self)
        self._pipeline_worker.finished_ok.connect(self._on_pipeline_done)
        self._pipeline_worker.failed.connect(self._on_pipeline_failed)
        self._pipeline_worker.start()

    def _on_arc_complete(self):
        # If recording is still finishing behind the scenes, make it explicit to the user.
        if not self.recording_done:
            self.status_label.setText("Finishing capture…")
            return
        # Keep disabled while either step is running
        if self.is_pipeline_running or self.is_classifying:
            return
        # Otherwise, re-enable controls
        self.btn_random.setEnabled(True)
        self.btn_start.setEnabled(self.session is not None and self.current_movement is not None)
        if self.status_label.text().startswith("Recording…"):
            self.status_label.setText("Recording complete.")

    # ----- Pipeline step callbacks -----
    def _on_pipeline_done(self, processed_tuple):
        self.is_pipeline_running = False

        # If EMG not selected, or not a proper tuple, skip classification.
        if not self.use_emg or not isinstance(processed_tuple, tuple) or len(processed_tuple) != 2:
            self.status_label.setText("Pipeline complete.")
            self.btn_random.setEnabled(True)
            self.btn_start.setEnabled(self.session is not None and self.current_movement is not None)
            return

        _, processed_emg_inlabel = processed_tuple  # classify IN-LABEL ONLY

        # Start CLASSIFICATION step with in-label windows only
        self.is_classifying = True
        self.status_label.setText("Classifying...")

        self._classification_worker = ClassificationWorker(processed_emg_inlabel, parent=self)
        self._classification_worker.finished_ok.connect(self._on_classification_done)
        self._classification_worker.failed.connect(self._on_classification_failed)
        self._classification_worker.start()

    def _on_pipeline_failed(self, msg: str):
        self.is_pipeline_running = False
        self.status_label.setText(f"Pipeline error: {msg}")
        self.btn_random.setEnabled(True)
        self.btn_start.setEnabled(self.session is not None and self.current_movement is not None)

    # ----- Classification step callbacks -----
    def _on_classification_done(self, result):
        self.is_classifying = False
        # Expect (A, B) from classify_emg() on IN-LABEL subset
        try:
            A, B = result
            def fmt_conf(x):
                conf = float(x)
                if conf <= 1.0:
                    conf *= 100.0  # accept probabilities in 0..1
                return f"{conf:.1f}%"

            true_id = self.params.get("movement_id", "N/A")
            if A == "":
                self.results_label.setText("In-label segment: no windows available for classification")
            else:
                self.results_label.setText(
                    f"True class {true_id} | In-label prediction: Class {A}, at {fmt_conf(B)} confidence"
                )
        except Exception:
            self.results_label.setText("Classification Results: placeholder")

        self.status_label.setText("Classification complete.")
        self.btn_random.setEnabled(True)
        self.btn_start.setEnabled(self.session is not None and self.current_movement is not None)

    def _on_classification_failed(self, msg: str):
        self.is_classifying = False
        self.status_label.setText(f"Classification error: {msg}")
        self.btn_random.setEnabled(True)
        self.btn_start.setEnabled(self.session is not None and self.current_movement is not None)

    # ---- Windowing helpers -------------------------------------------------

    def _get_fs(self, kind: str) -> float:
        """
        Sampling rate from session.config.
        kind: "emg" or "eeg"
        """
        cfg = getattr(self.session, "config", None)
        if cfg is None:
            return 2000.0
        try:
            if kind.lower() == "emg":
                fs = float(getattr(cfg, "EMG_FS"))
            else:
                fs = float(getattr(cfg, "EEG_FS"))
            return fs if fs > 0 else 1000.0
        except Exception:
            return 1000.0

    def _ms_to_samples(self, ms: float, fs: float) -> int:
        return max(1, int(round(ms * fs / 1000.0)))

    def _window_by_samples_fallback(self, data_2d: np.ndarray, win: int, overlap: int) -> np.ndarray:
        """
        Fallback windowing: returns array shape (n_windows, win, n_channels).
        Expects data_2d shape (samples, channels).
        """
        if data_2d.ndim != 2:
            raise ValueError("Expected 2D array (samples, channels).")
        n, c = data_2d.shape
        step = max(1, win - overlap)
        if win <= 0 or step <= 0 or win > n:
            return np.empty((0, win, c))
        starts = range(0, n - win + 1, step)
        windows = [data_2d[s:s+win, :] for s in starts]
        if not windows:
            return np.empty((0, win, c))
        return np.stack(windows, axis=0)

    def _call_window_data_safe(self, arr: np.ndarray, win_samp: int, overlap_samp: int) -> np.ndarray:
        """
        Call the imported window_data with likely signatures, or fall back to a local implementation.
        Expects arr shape (samples, channels).
        """
        # Try kwargs with common names
        try:
            return window_data(arr, window_size=win_samp, overlap=overlap_samp)
        except TypeError:
            pass
        # Try positional (arr, window_size, overlap)
        try:
            return window_data(arr, win_samp, overlap_samp)
        except TypeError:
            pass
        # Fallback
        return self._window_by_samples_fallback(arr, win_samp, overlap_samp)

    # ---- Pipeline ----------------------------------------------------------

    def run_pipeline(self, movement_name: str, data):
        """
        Returns:
            (processed_emg_all, processed_emg_inlabel)
              processed_emg_all      (np.ndarray | None): windowed + (optionally) normalised EMG (all samples)
              processed_emg_inlabel  (np.ndarray | None): windowed + (optionally) normalised EMG built from a
                                                          *separately processed* raw array that contains only
                                                          samples whose label is non-zero (i.e., no zeros).
        """
        print(f"[processing] movement: {movement_name}")
        print(f"[processing] data shape: {getattr(data, 'shape', None)}")

        os.makedirs("data", exist_ok=True)

        emg_data = data[self.session.config.MUOVI_EMG_CHANNELS] if getattr(self.session.config, "USE_EMG",
                                                                           False) else None
        eeg_data = data[self.session.config.MUOVI_PLUS_EEG_CHANNELS] if getattr(self.session.config, "USE_EEG",
                                                                                False) else None

        # movement_id is stored on selection (1..N)
        movement_id = int(self.params.get("movement_id", -1))

        # Build label vector (samples-long) with the movement_id, or use auto segmentation if EMG is present and enabled
        if self.params["use_auto"] and emg_data is not None:
            label = detect_movement_mask(emg_data)
            label_type = "auto"
        else:
            label = np.full(data.shape[1], movement_id, dtype=float)
            label_type = "basic"

        processed_emg_all = None
        processed_emg_inlabel = None

        if SAVE_AS_NPY:
            np.save("data/online_data.npy", data)
            print("[processing] saved trial to data/online_data.npy")
        else:
            np.savetxt(
                f"data/trial_{self.params['trial']}_{label_type}_label.csv",
                label.reshape(-1, 1), delimiter=","
            )

            # ---- EMG branch ----------------------------------------------------
            if getattr(self.session.config, "USE_EMG", False) and emg_data is not None:
                # emg_data is channels x samples; save raw then ensure (samples, channels)
                np.savetxt(f"data/trial_{self.params['trial']}_raw_emg.csv",
                           emg_data.transpose(), delimiter=",")
                # Ensure shape (samples, channels)
                if emg_data.shape[0] < emg_data.shape[1]:
                    emg_data = emg_data.T  # now (samples, channels)

                # --- common window params
                fs_emg = self._get_fs("emg")
                win_emg = self._ms_to_samples(self.params["window_ms"], fs_emg)
                ov_emg = self._ms_to_samples(self.params["overlap_ms"], fs_emg)
                ov_emg = min(ov_emg, max(0, win_emg - 1))  # enforce 0 <= overlap < window

                # --- FULL EMG: filter -> normalise -> window
                filtered_all = selective_filter(self.params["filters"]["emg"], emg_data)
                normalised_all = normalise_data(filtered_all) if self.params["use_normalisation"] else filtered_all
                windowed_all = self._call_window_data_safe(normalised_all, win_emg, ov_emg)
                if getattr(windowed_all, "ndim", 0) != 3:
                    print("Warning: EMG windowing produced unexpected shape; skipping classification data (all).")
                    processed_emg_all = None
                else:
                    processed_emg_all = windowed_all

                # --- IN-LABEL EMG: build a new raw array with ONLY non-zero labels, then process separately
                segs = mask_to_segments(label)
                if len(segs) == 0:
                    # no non-zero segments
                    emg_data_inlabel = emg_data[0:0, :]
                else:
                    # take the first non-zero segment (start..end inclusive)
                    start, end = segs[0]
                    # end is inclusive in mask_to_segments; slice with end+1
                    emg_data_inlabel = emg_data[start:end+1, :]

                if emg_data_inlabel.shape[0] >= win_emg and emg_data_inlabel.size > 0:
                    filtered_inlabel = selective_filter(self.params["filters"]["emg"], emg_data_inlabel)
                    normalised_inlabel = normalise_data(filtered_inlabel) if self.params["use_normalisation"] else filtered_inlabel
                    windowed_inlabel = self._call_window_data_safe(normalised_inlabel, win_emg, ov_emg)
                    if getattr(windowed_inlabel, "ndim", 0) == 3:
                        processed_emg_inlabel = windowed_inlabel
                    else:
                        print("Warning: EMG windowing (in-label) produced unexpected shape.")
                        processed_emg_inlabel = np.empty((0, win_emg, emg_data.shape[1]))
                else:
                    # Not enough non-zero samples to form at least one full window
                    processed_emg_inlabel = np.empty((0, win_emg, emg_data.shape[1]))

                # Save both sets
                with h5py.File(f"data/trial_{self.params['trial']}_processed_emg.h5", "w") as f:
                    if processed_emg_all is not None:
                        f.create_dataset("windowed_data", data=processed_emg_all)
                    if processed_emg_inlabel is not None:
                        f.create_dataset("windowed_data_inlabel", data=processed_emg_inlabel)

            # ---- EEG branch ----------------------------------------------------
            if getattr(self.session.config, "USE_EEG", False) and eeg_data is not None:
                np.savetxt(f"data/trial_{self.params['trial']}_raw_eeg.csv",
                           eeg_data.transpose(), delimiter=",")
                if eeg_data.shape[0] < eeg_data.shape[1]:
                    eeg_data = eeg_data.T
                reduced = reduce_eeg(eeg_data)
                filtered = selective_filter(self.params["filters"]["eeg"], reduced)
                normalised = normalise_data(filtered) if self.params["use_normalisation"] else filtered

                fs_eeg = self._get_fs("eeg")
                win_eeg = self._ms_to_samples(self.params["window_ms"], fs_eeg)
                ov_eeg = self._ms_to_samples(self.params["overlap_ms"], fs_eeg)
                ov_eeg = min(ov_eeg, max(0, win_eeg - 1))

                windowed = self._call_window_data_safe(normalised, win_eeg, ov_eeg)
                if getattr(windowed, "ndim", 0) != 3:
                    print("Warning: EEG windowing produced unexpected shape.")
                with h5py.File(f"data/trial_{self.params['trial']}_processed_eeg.h5", "w") as f:
                    f.create_dataset("windowed_data", data=windowed)

        print("[processing] pipeline completed")
        if processed_emg_all is not None:
            print(f"[processing] processed EMG (all) shape: {processed_emg_all.shape}")
        if processed_emg_inlabel is not None:
            print(f"[processing] processed EMG (in-label) shape: {processed_emg_inlabel.shape}")
        return (processed_emg_all, processed_emg_inlabel)

    def closeEvent(self, event):
        try:
            if self._flusher is not None and self._flusher.isRunning():
                self._flusher.stop()
                self._flusher.wait(500)
            if self.session is not None:
                self.session.finish()
        except Exception:
            pass
        super().closeEvent(event)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Online Study")
        self.resize(1000, 720)  # a bit larger to give the image more space
        self.stack = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stack)

        self.params: Dict[str, Any] = {}
        self.use_emg = True
        self.use_eeg = True

        # Page 1: device select
        self.page_select = DeviceSelectPage()
        self.page_select.proceed.connect(self._go_params)
        self.stack.addWidget(self.page_select)

    # Flow: DeviceSelect -> Parameters -> Experiment
    def _go_params(self, use_emg: bool, use_eeg: bool):
        self.use_emg, self.use_eeg = use_emg, use_eeg
        self.page_params = ParametersPage(use_emg=self.use_emg, use_eeg=self.use_eeg)
        self.page_params.proceed.connect(self._go_experiment)
        self.page_params.btn_back.clicked.connect(lambda: self.stack.setCurrentWidget(self.page_select))
        self.stack.addWidget(self.page_params)
        self.stack.setCurrentWidget(self.page_params)

    def _go_experiment(self, params: dict):
        self.params = params
        self.page_experiment = ExperimentPage(self.use_emg, self.use_eeg, params=self.params)
        self.stack.addWidget(self.page_experiment)
        self.stack.setCurrentWidget(self.page_experiment)

    def closeEvent(self, event):
        try:
            if hasattr(self, "page_experiment"):
                # ensure flusher stops and session closes
                self.page_experiment.close()
        except Exception:
            pass
        super().closeEvent(event)


def mask_to_segments(mask: np.ndarray):
    """
    Convert a 1D binary mask into start/end index pairs
    for contiguous non-zero regions.

    Args:
        mask: 1D array-like of 0/1 values

    Returns:
        List of (start, end) index tuples (end is inclusive)
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    # Find rising and falling edges
    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends   = np.where(diff == -1)[0]

    # Handle edge cases if mask starts/ends inside a region
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, len(mask) - 1]

    return list(zip(starts, ends))


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
