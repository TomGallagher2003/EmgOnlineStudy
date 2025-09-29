# movement_timer.py
# PyQt5 app: device select -> parameter entry -> device check/init -> random movement -> 4s arc timer -> run_pipeline()
# - Device check/init happens right after parameters screen (session kept open)
# - 4s recording runs in background
# - After recording, pipeline runs in background; buttons stay disabled and status shows progress messages

import os
import sys
import random
from typing import Optional, Dict, Any

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from util.filters import selective_filter
from util.images import Images
from recording import Session  # adjust import path if needed
from util.normalise_data import normalise_data
from util.windows import window_data

# Expecting util.images to expose MOVEMENT_TUPLES = list[(clean_name, filename)]
MOVEMENTS = Images.MOVEMENT_TUPLES

# Faster than CSV if you want speedier saves inside run_pipeline
SAVE_AS_NPY = False


class ArcTimerWidget(QtWidgets.QWidget):
    finished = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._progress = 0.0
        self._duration_ms = 4000
        self._tick_ms = 30
        self._elapsed = 0
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self.setMinimumSize(160, 160)

    def start(self, duration_ms=4000):
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
    """Collect trial number, recording length, and filter A/B/C choices.
    We only validate and store these; pipeline doesn't use them yet.
    """

    proceed = QtCore.pyqtSignal(dict)

    FILTER_OPTIONS = [
        "None",
        "Highpass 0.3 Hz",
        "Notch 50 Hz",
        "Notch 100 Hz",
        "Lowpass 70 Hz",
    ]

    def __init__(self, parent=None):
        super().__init__(parent)

        title = QtWidgets.QLabel("Experiment Parameters")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: 600;")

        # Trial number
        self.trial_edit = QtWidgets.QLineEdit()
        self.trial_edit.setPlaceholderText("e.g., 1")

        # Recording length (seconds)
        self.length_edit = QtWidgets.QLineEdit()
        self.length_edit.setPlaceholderText("seconds, e.g., 4.0")

        # Filters A/B/C
        self.filter_a = QtWidgets.QComboBox(); self.filter_a.addItems(self.FILTER_OPTIONS)
        self.filter_b = QtWidgets.QComboBox(); self.filter_b.addItems(self.FILTER_OPTIONS)
        self.filter_c = QtWidgets.QComboBox(); self.filter_c.addItems(self.FILTER_OPTIONS)

        form = QtWidgets.QFormLayout()
        form.addRow("Trial number:", self.trial_edit)
        form.addRow("Recording length (s):", self.length_edit)
        form.addRow("Filter A:", self.filter_a)
        form.addRow("Filter B:", self.filter_b)
        form.addRow("Filter C:", self.filter_c)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_back = QtWidgets.QPushButton("Back")
        self.btn_next = QtWidgets.QPushButton("Continue")
        self.btn_next.setDefault(True)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_back)
        btn_row.addWidget(self.btn_next)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addSpacing(12)
        layout.addWidget(title)
        layout.addSpacing(8)
        layout.addLayout(form)
        layout.addStretch(1)
        layout.addLayout(btn_row)

        self.btn_next.clicked.connect(self._on_continue)

    def _on_continue(self):
        # Validate: trial must be int; recording length must be number
        trial_text = self.trial_edit.text().strip()
        length_text = self.length_edit.text().strip()

        try:
            trial_num = int(trial_text)
        except Exception:
            QtWidgets.QMessageBox.warning(self, "Invalid trial number", "Trial number must be an integer.")
            return

        try:
            rec_len = float(length_text)
        except Exception:
            QtWidgets.QMessageBox.warning(self, "Invalid recording length", "Recording length must be a number (seconds).")
            return

        if rec_len <= 0:
            QtWidgets.QMessageBox.warning(self, "Invalid recording length", "Recording length must be > 0.")
            return

        params = {
            "trial": trial_num,
            "recording_length": rec_len,
            "filter_a": self.filter_a.currentText(),
            "filter_b": self.filter_b.currentText(),
            "filter_c": self.filter_c.currentText(),
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
                _ = session.get_record(rec_time=0.4)
            except Exception:
                self.failed.emit("Device check failed. Please reboot devices and try again.")
                return

            self.ready.emit(session)

        except Exception:
            self.failed.emit("Device init failed. Please reboot devices and try again.")


class RecordingWorker(QtCore.QThread):
    finished_ok = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)
    capture_started = QtCore.pyqtSignal()

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session

    def run(self):
        try:
            self.capture_started.emit()
            # NOTE: still hard-coded 4.0s capture; we are not yet using the user-set recording length
            data = self.session.get_record(rec_time=4.0)
            if data is None or data.size == 0:
                self.failed.emit("Recording returned no data.")
                return
            self.finished_ok.emit(data)
        except Exception as e:
            self.failed.emit(str(e))


class ClassificationWorker(QtCore.QThread):
    finished_ok = QtCore.pyqtSignal()
    failed = QtCore.pyqtSignal(str)

    def __init__(self, classify_fn, movement_name, data, parent=None):
        super().__init__(parent)
        self.classify_fn = classify_fn
        self.movement_name = movement_name
        self.data = data

    def run(self):
        try:
            self.classify_fn(self.movement_name, self.data)
            self.finished_ok.emit()
        except Exception as e:
            self.failed.emit(str(e))


class ExperimentPage(QtWidgets.QWidget):
    """Random movement + Start (4s) — session is initialized on entry (device check happens here)."""

    def __init__(self, use_emg, use_eeg, params: Optional[Dict[str, Any]] = None, parent=None):
        super().__init__(parent)
        self.use_emg = use_emg
        self.use_eeg = use_eeg
        self.params = params or {}
        self.current_movement = None
        self.session: Optional[Session] = None
        self.recording_worker = None
        self.is_classifying = False
        self.recording_done = False  # track when recording thread is actually done

        # UI
        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setMinimumHeight(220)
        self.image_label.setStyleSheet("background: #fafafa; border: 1px solid #e6e6e6;")

        self.name_label = QtWidgets.QLabel("No movement selected")
        self.name_label.setAlignment(QtCore.Qt.AlignCenter)
        self.name_label.setStyleSheet("font-size: 18px;")

        self.status_label = QtWidgets.QLabel("Initializing devices…")
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.status_label.setStyleSheet("color:#666;")

        self.btn_random = QtWidgets.QPushButton("Get Random Movement")
        self.btn_random.clicked.connect(self.pick_random_movement)
        self.btn_random.setEnabled(False)  # disabled until devices ready

        self.btn_start = QtWidgets.QPushButton("Start Recording (4s)")
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self.start_recording)

        self.arc = ArcTimerWidget()
        self.arc.finished.connect(self._on_arc_complete)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.image_label)
        layout.addSpacing(8)
        layout.addWidget(self.name_label)
        layout.addWidget(self.status_label)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.btn_random)
        row.addWidget(self.btn_start)
        layout.addLayout(row)
        layout.addWidget(self.arc, 1)
        layout.addStretch(1)

        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        # Kick off device init immediately
        self._init_worker = DeviceInitWorker(self.use_emg, self.use_eeg, parent=self)
        self._init_worker.ready.connect(self._on_devices_ready)
        self._init_worker.failed.connect(self._on_devices_failed)
        self._init_worker.start()

    # ----- Device init callbacks -----
    def _on_devices_ready(self, session: Session):
        self.session = session
        self.status_label.setText("Devices ready.")
        self.btn_random.setEnabled(True)
        self.btn_start.setEnabled(self.current_movement is not None)

    def _on_devices_failed(self, msg: str):
        self.status_label.setText(msg)

    # ----- Experiment flow -----
    def pick_random_movement(self):
        self.current_movement = random.choice(MOVEMENTS)
        name, filename = self.current_movement
        self.name_label.setText(name)
        img_path = os.path.join(self.script_dir, filename)
        if os.path.exists(img_path):
            pix = QtGui.QPixmap(img_path)
            if not pix.isNull():
                scaled = pix.scaled(
                    self.image_label.size() * 0.95,
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation,
                )
                self.image_label.setPixmap(scaled)
            else:
                self.image_label.clear()
        else:
            self.image_label.clear()

        # Enable start if devices ready
        if self.session is not None:
            self.btn_start.setEnabled(True)

    def start_recording(self):
        if self.arc.is_running() or self.session is None:
            return
        if not self.current_movement:
            QtWidgets.QMessageBox.information(self, "Pick a movement", "Please choose a movement first.")
            return

        self.status_label.setText("Recording…")
        self.btn_random.setEnabled(False)
        self.btn_start.setEnabled(False)
        self.recording_done = False

        self.recording_worker = RecordingWorker(self.session, parent=self)
        self.recording_worker.capture_started.connect(lambda: self.arc.start(4000))  # start arc on actual capture start
        self.recording_worker.finished_ok.connect(self._on_recording_finished)
        self.recording_worker.failed.connect(self._on_recording_failed)
        self.recording_worker.start()

    def _on_recording_failed(self, msg: str):
        self.status_label.setText(f"Error: {msg}")
        self.btn_random.setEnabled(True)
        self.btn_start.setEnabled(True)
        self.arc.stop()
        self.recording_done = True

    def _on_recording_finished(self, data):
        self.recording_done = True
        # Start pipeline immediately when data arrives (buttons remain disabled)
        movement_name = self.current_movement[0] if self.current_movement else None

        self.is_classifying = True
        self.btn_random.setEnabled(False)
        self.btn_start.setEnabled(False)
        self.status_label.setText("Classifying data...")

        self._clf_worker = ClassificationWorker(self.run_pipeline, movement_name, data, parent=self)
        self._clf_worker.finished_ok.connect(self._on_classification_done)
        self._clf_worker.failed.connect(self._on_classification_failed)
        self._clf_worker.start()

    def _on_arc_complete(self):
        # If recording is still finishing behind the scenes, make it explicit to the user.
        if not self.recording_done:
            self.status_label.setText("Finishing capture…")
            return

        # If recording finished but we are still classifying, keep disabled
        if self.is_classifying:
            return

        # Otherwise, re-enable controls (e.g., if something finished very fast)
        self.btn_random.setEnabled(True)
        self.btn_start.setEnabled(self.session is not None and self.current_movement is not None)
        if self.status_label.text().startswith("Recording…"):
            self.status_label.setText("Recording complete.")

    def _on_classification_done(self):
        self.is_classifying = False
        self.status_label.setText("Classification complete.")
        self.btn_random.setEnabled(True)
        self.btn_start.setEnabled(self.session is not None and self.current_movement is not None)

    def _on_classification_failed(self, msg: str):
        self.is_classifying = False
        self.status_label.setText(f"Classification error: {msg}")
        self.btn_random.setEnabled(True)
        self.btn_start.setEnabled(self.session is not None and self.current_movement is not None)

    def run_pipeline(self, movement_name: str, data):
        print(f"[processing] movement: {movement_name}")
        print(f"[processing] data shape: {getattr(data, 'shape', None)}")

        os.makedirs("data", exist_ok=True)

        # Save for inspection (CSV default; toggle to NPY for speed)
        if SAVE_AS_NPY:
            np.save("data/online_data.npy", data)
            print("[processing] saved trial to data/online_data.npy")
        else:
            if getattr(self.session.config, "USE_EMG", False):
                np.savetxt(f"data/trial_{self.params['trial']}_raw_emg.csv", data[self.session.config.MUOVI_EMG_CHANNELS].transpose(), delimiter=",")
            if getattr(self.session.config, "USE_EEG", False):
                np.savetxt(f"data/trial_{self.params['trial']}_raw_eeg.csv", data[self.session.config.MUOVI_PLUS_EEG_CHANNELS].transpose(), delimiter=",")
            print("Saved raw trial CSVs.")

        filtered_data = selective_filter([self.params["filter_a"], self.params["filter_b"], self.params["filter_c"]], data)
        windowed_data = window_data(filtered_data)
        normalised_data = normalise_data(windowed_data)

        if SAVE_AS_NPY:
            np.save("data/processed.npy", normalised_data)
            print("[processing] Saved processed to data/processed.npy")
        else:
            if getattr(self.session.config, "USE_EMG", False):
                np.savetxt(f"data/trial_{self.params['trial']}_processed_emg.csv", normalised_data[self.session.config.MUOVI_EMG_CHANNELS].transpose(), delimiter=",")
            if getattr(self.session.config, "USE_EEG", False):
                np.savetxt(f"data/trial_{self.params['trial']}_processed_eeg.csv", normalised_data[self.session.config.MUOVI_PLUS_EEG_CHANNELS].transpose(), delimiter=",")
            print("[processing] saved processed CSVs.")

        print("[processing] pipeline completed")

    def closeEvent(self, event):
        try:
            if self.session is not None:
                self.session.finish()
        except Exception:
            pass
        super().closeEvent(event)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Movement Timer & Classifier")
        self.resize(900, 640)
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
        self.page_params = ParametersPage()
        self.page_params.proceed.connect(self._go_experiment)
        # Optional: allow going back
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
            if hasattr(self, "page_experiment") and self.page_experiment.session is not None:
                self.page_experiment.session.finish()
        except Exception:
            pass
        super().closeEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
