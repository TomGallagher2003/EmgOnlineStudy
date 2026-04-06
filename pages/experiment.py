import os
import pathlib
import random
from typing import Optional, Dict, Any

import h5py
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from pipeline_sections.filters import selective_filter
from pipeline_sections.normalisation import normalise_data
from pipeline_sections.reduce_eeg_samples import reduce_eeg
from pipeline_sections.windows import window_data, window_labels
from util.images import Images
from util.mask_to_segments import mask_to_segments
from util.movement_segmentation import detect_movement_mask
from util.recording import Session
from widgets.arc_timer import ArcTimerWidget
from workers.classification import ClassificationWorker, FusionClassificationWorker
from workers.device_init import DeviceInitWorker
from workers.flush import FlushWorker
from workers.pipeline import PipelineWorker
from workers.recording import RecordingWorker
from pipeline_sections.models.full_training import process_h5_files, evaluate_model, EMGDataset, CNN1D_Transformer, CNN1D, TransformerModel
from pipeline_sections.models.evaluation import ChannelAdapter, CNN1D, CNN1D_Transformer, TransformerModel, EMGDataset, DataLoader
from main_settings import MODEL_PATH, EEG_MODEL_PATH, FUSION_MODEL_PATH, MODEL_MODE

"""Experiment execution page (GUI).

This module defines :class:`ExperimentPage`, a PyQt5 widget that orchestrates a
single-trial workflow:

1) Device init/check and session creation on entry.
2) Random movement selection with large preview image.
3) Timed recording (Arc timer) on a background thread.
4) Post-recording pipeline (filter → normalise → window) for EMG/EEG.
5) Optional EMG-only classification using H5 written during processing.

Data persisted per trial:
    data/trial_<trial_num>/rec_<N>/
        - raw_emg.csv, raw_emg.h5 (if EMG enabled)
        - processed_emg.h5 with windowed data + labels
        - raw_eeg.csv, processed_eeg.h5 (if EEG enabled)
        - label.csv
        - classification_report.txt (after classification)

Notes:
    - Imports intentionally re-export model symbols required elsewhere. Do not
      remove imports.
    - No behavior changes; only documentation added for mkdocstrings.
"""

MOVEMENTS = Images.MOVEMENT_TUPLES

SAVE_AS_NPY = False


class ExperimentPage(QtWidgets.QWidget):
    """Random movement + Start — session is initialized on entry (device check happens here).

    This widget manages the full "one attempt" flow: pick a random movement, run a
    fixed-length capture, process the data, and (if EMG is enabled) run a quick
    in-label classification using the H5 artifacts produced by the pipeline.

    Args:
        use_emg: Whether EMG channels are active for this experiment.
        use_eeg: Whether EEG channels are active for this experiment.
        params: Optional runtime parameters dict. Expected keys include
            - ``recording_length`` (float, seconds)
            - ``window_ms`` (int)
            - ``overlap_ms`` (int)
            - ``use_auto`` (bool): use auto segmentation for labels when EMG present
            - ``use_normalisation`` (bool)
            - ``filters`` (dict): e.g., ``{"emg": {...}, "eeg": {...}}``
            - ``trial`` (int): current trial number (used in folder naming)
        parent: Optional Qt parent.

    Signals:
        (Signals are on the various worker threads; connect within this widget.)
    """

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

        # NEW: remember the current recording directory for this trial
        self._current_rec_dir: Optional[str] = None

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
        self.results_label.setMinimumHeight(36)
        self.results_label.setStyleSheet("font-size: 30px;")
        rb_layout.addWidget(self.results_label)
        self.results_box.setVisible(self.use_emg or self.use_eeg)

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

        self.images_root = pathlib.Path(__file__).resolve().parents[1]

        # NEW: default model/report paths
        self.project_root = self.images_root
        self.model_path = str(self.project_root / "pipeline_sections" / "models" / "model.pth")

        # Kick off device init immediately
        self._init_worker = DeviceInitWorker(self.use_emg, self.use_eeg, parent=self)
        self._init_worker.ready.connect(self._on_devices_ready)
        self._init_worker.failed.connect(self._on_devices_failed)
        self._init_worker.start()

    # --------------------- NEW: per-trial, incrementing recording folders ---------------------
    def _prepare_recording_dir(self, trial_num: int) -> str:
        """
        Create and return a unique recording directory for this trial.

        The path format is:

            ``data/trial_<trial_num>/rec_<N>/``

        where ``N`` is a 1-based counter incremented for each capture within the
        same trial.

        Args:
            trial_num: Trial identifier used to group multiple recordings.

        Returns:
            Absolute or relative directory path created for the next recording.
        """
        base_dir = os.path.join("data", f"trial_{trial_num}")
        os.makedirs(base_dir, exist_ok=True)

        # find existing rec_* dirs and pick next N
        existing = []
        for name in os.listdir(base_dir):
            if name.startswith("rec_"):
                try:
                    existing.append(int(name.split("_", 1)[1]))
                except Exception:
                    pass
        next_n = (max(existing) + 1) if existing else 1

        rec_dir = os.path.join(base_dir, f"rec_{next_n}")
        os.makedirs(rec_dir, exist_ok=False)
        print(f"[save] trial {trial_num} -> {rec_dir}")
        return rec_dir
    # ------------------------------------------------------------------------------------------

    # scale image on resize
    def resizeEvent(self, event):
        """Keep the preview image scaled to the label size on widget resize."""
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
        """Start the idle flusher thread if a session exists and no flusher is running."""
        if self.session is None:
            return
        if self._flusher is not None and self._flusher.isRunning():
            return
        self._flusher = FlushWorker(self.session, chunk_sec=0.5, parent=self)
        self._flusher.start()
        print("[flush] started")

    def _stop_flusher(self):
        """Request the idle flusher to stop and wait briefly for termination."""
        if self._flusher is not None:
            self._flusher.stop()
            # wait briefly to ensure it stops before recording starts
            self._flusher.wait(500)
            print("[flush] stopped")

    # ----- Device init callbacks -----
    def _on_devices_ready(self, session: Session):
        """UI handler invoked when devices are initialized and the session is ready."""
        self.session = session
        self.status_label.setText("Devices ready.")
        self.btn_random.setEnabled(True)
        self.btn_start.setEnabled(self.current_movement is not None)
        # start background flushing as soon as session is open
        self._start_flusher()

    def _on_devices_failed(self, msg: str):
        """UI handler invoked on device/session initialization failure."""
        self.status_label.setText(msg)

    # ----- Experiment flow -----
    def pick_random_movement(self):
        """Select a random movement (avoiding immediate repeats) and update the preview image."""
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
        img_path = os.path.join(self.images_root, filename)
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
        """Begin a timed recording for the selected movement and hook up callbacks."""
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
            self.results_label.setText("Classification Results:")

        self.recording_worker = RecordingWorker(self.session, self.params["recording_length"], parent=self)
        # start arc on actual capture start; duration follows user-set recording length
        self.recording_worker.capture_started.connect(lambda: self.arc.start(int(self.params["recording_length"] * 1000)))
        self.recording_worker.finished_ok.connect(self._on_recording_finished)
        self.recording_worker.failed.connect(self._on_recording_failed)
        self.recording_worker.start()

    def _on_recording_failed(self, msg: str):
        """Handle capture failure: surface error, re-enable controls, and resume flushing."""
        self.status_label.setText(f"Error: {msg}")
        self.btn_random.setEnabled(True)
        self.btn_start.setEnabled(True)
        self.arc.stop()
        self.recording_done = True
        # Resume flushing since we're idle again
        self._start_flusher()

    def _on_recording_finished(self, data):
        """Handle successful capture by starting the processing pipeline."""
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
        """When the arc timer finishes, restore controls if no background work remains."""
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
        """Handle pipeline completion and begin classification based on MODEL_MODE."""
        self.is_pipeline_running = False

        can_classify_emg = self.use_emg and MODEL_MODE in ("emg", "fusion")
        can_classify_eeg = self.use_eeg and MODEL_MODE in ("eeg", "fusion")
        can_classify = can_classify_emg or can_classify_eeg

        if not can_classify or not isinstance(processed_tuple, tuple):
            self.status_label.setText("Pipeline complete.")
            self.btn_random.setEnabled(True)
            self.btn_start.setEnabled(self.session is not None and self.current_movement is not None)
            return

        self.is_classifying = True
        self.status_label.setText("Classifying...")

        fs_emg = self._get_fs("emg")
        sample_size = self._ms_to_samples(self.params["window_ms"], fs_emg)
        report_path = os.path.join(self._current_rec_dir, "classification_report.txt")
        folder = self._current_rec_dir or ""

        if MODEL_MODE == "fusion":
            self._classification_worker = FusionClassificationWorker(
                folder_path=folder,
                model_path=FUSION_MODEL_PATH,
                report_save_path=report_path,
                sample_size=sample_size,
                batch_size=512,
                num_classes=30,
                parent=self,
            )
        elif MODEL_MODE == "eeg":
            self._classification_worker = ClassificationWorker(
                folder_path=folder,
                model_path=EEG_MODEL_PATH,
                report_save_path=report_path,
                sample_size=sample_size,
                batch_size=512,
                num_classes=30,
                repeats=1,
                parent=self,
            )
        else:  # "emg"
            self._classification_worker = ClassificationWorker(
                folder_path=folder,
                model_path=MODEL_PATH,
                report_save_path=report_path,
                sample_size=sample_size,
                batch_size=512,
                num_classes=30,
                repeats=1,
                parent=self,
            )

        self._classification_worker.finished_ok.connect(self._on_classification_done)
        self._classification_worker.failed.connect(self._on_classification_failed)
        self._classification_worker.start()

    def _on_pipeline_failed(self, msg: str):
        """Handle pipeline failure and restore controls."""
        self.is_pipeline_running = False
        self.status_label.setText(f"Pipeline error: {msg}")
        self.btn_random.setEnabled(True)
        self.btn_start.setEnabled(self.session is not None and self.current_movement is not None)

    # ----- Classification step callbacks -----
    def _on_classification_done(self, result):
        """Render classification result tuple ``(pred_class, confidence)`` to the UI."""
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
                self.results_label.setText("No windows available for classification")
            else:
                self.results_label.setText(
                    f"True class {true_id} | Prediction: Class {A}, at {fmt_conf(B)} confidence"
                )
        except Exception:
            self.results_label.setText("Classification Results:")

        self.status_label.setText("Classification complete.")
        self.btn_random.setEnabled(True)
        self.btn_start.setEnabled(self.session is not None and self.current_movement is not None)

    def _on_classification_failed(self, msg: str):
        """Handle classification error and restore controls."""
        self.is_classifying = False
        self.status_label.setText(f"Classification error: {msg}")
        self.btn_random.setEnabled(True)
        self.btn_start.setEnabled(self.session is not None and self.current_movement is not None)

    # ---- Windowing helpers -------------------------------------------------

    def _get_fs(self, kind: str) -> float:
        """
        Get sampling rate from ``session.config``.

        Args:
            kind: Either ``"emg"`` or ``"eeg"``.

        Returns:
            Sampling frequency in Hz. Falls back to sensible defaults if the
            configuration is missing.
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
        """Convert milliseconds to integer sample count at sampling rate ``fs``."""
        return max(1, int(round(ms * fs / 1000.0)))

    def _window_by_samples_fallback(self, data_2d: np.ndarray, win: int, overlap: int) -> np.ndarray:
        """
        Fallback windowing routine.

        Builds windows of shape ``(n_windows, win, n_channels)`` from a 2-D
        array of shape ``(samples, channels)`` using step size
        ``max(1, win - overlap)``.

        Args:
            data_2d: Input array, 2-D (samples, channels).
            win: Window length in samples.
            overlap: Overlap in samples (``0 <= overlap < win``).

        Returns:
            Windowed array of shape ``(n_windows, win, n_channels)`` or an
            empty array if parameters are invalid or not enough samples.
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
        Call :func:`window_data` with common signatures; fallback locally if needed.

        Attempts:
            1) ``window_data(arr, window_size=win_samp, overlap=overlap_samp)``
            2) ``window_data(arr, win_samp, overlap_samp)``
            3) Local fallback :meth:`_window_by_samples_fallback`.

        Args:
            arr: Array of shape ``(samples, channels)``.
            win_samp: Window length in samples.
            overlap_samp: Overlap in samples.

        Returns:
            Windowed array of shape ``(n_windows, win, channels)`` (possibly empty).
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
        Run the online processing pipeline and persist artifacts.

        Steps (EMG branch):
            1) Build label vector (auto segmentation if enabled & EMG present).
            2) Save raw and label artifacts to per-trial/recording folder.
            3) Filter → (optional) normalise → window (all samples).
            4) Build in-label-only raw buffer (non-zero labels), then
               filter/normalise/window separately for classification.

        Steps (EEG branch):
            1) Save raw.
            2) Reduce → filter → (optional) normalise → window.
            3) Save windowed EEG to H5.

        Args:
            movement_name: Name of the selected movement (for logging/metadata).
            data: Raw capture payload (array-like). EMG/EEG slices are derived
                using indices from ``session.config``.

        Returns:
            tuple:
                processed_emg_all (np.ndarray | None):
                    Windowed (+normalised if enabled) EMG for all samples.
                processed_emg_inlabel (np.ndarray | None):
                    Windowed (+normalised if enabled) EMG built from a
                    separately constructed raw buffer containing only non-zero
                    labels (i.e., no rest).

        Notes:
            - Persists multiple CSV/H5 files into ``data/trial_<t>/rec_<N>/``.
            - ``sample_size`` for classification is derived from ``window_ms``.
        """
        print(f"[processing] movement: {movement_name}")
        print(f"[processing] data shape: {getattr(data, 'shape', None)}")

        # --------- pick unique recording directory for this trial ---------
        os.makedirs("data", exist_ok=True)
        trial_num = int(self.params.get("trial", 0))
        rec_dir = self._prepare_recording_dir(trial_num)
        self._current_rec_dir = rec_dir  # remember for classification worker
        # ------------------------------------------------------------------

        emg_data = data[self.session.config.MUOVI_EMG_CHANNELS] if getattr(self.session.config, "USE_EMG",
                                                                           False) else None
        eeg_data = data[self.session.config.MUOVI_PLUS_EEG_CHANNELS] if getattr(self.session.config, "USE_EEG",
                                                                                False) else None

        # movement_id is stored on selection (1..N)
        movement_id = int(self.params.get("movement_id", -1))

        # Build label vector (samples-long) with the movement_id, or use auto segmentation if EMG is present and enabled
        if self.params["use_auto"] and emg_data is not None:
            label = detect_movement_mask(emg_data) * movement_id
        else:
            label = np.full(data.shape[1], movement_id, dtype=float)
        processed_emg_all = None
        processed_emg_inlabel = None

        if SAVE_AS_NPY:
            np.save(os.path.join(rec_dir, "online_data.npy"), data)
            print(f"[processing] saved trial to {os.path.join(rec_dir, 'online_data.npy')}")
        else:
            np.savetxt(
                os.path.join(rec_dir, "label.csv"),
                label.reshape(-1, 1), delimiter=","
            )

            # ---- EMG branch ----------------------------------------------------
            if getattr(self.session.config, "USE_EMG", False) and emg_data is not None:
                # emg_data is channels x samples; save raw then ensure (samples, channels)
                np.savetxt(os.path.join(rec_dir, "raw_emg.csv"),
                           emg_data.transpose(), delimiter=",")

                with h5py.File(os.path.join(rec_dir, "raw_emg.h5"), "w") as f:
                    if emg_data is not None:
                        f.create_dataset("emg", data=emg_data.transpose())
                    if label is not None:
                        f.create_dataset("label", data=label)

                # Ensure shape (samples, channels)
                if emg_data.shape[0] < emg_data.shape[1]:
                    emg_data = emg_data.T  # now (samples, channels)

                # --- common window params
                fs_emg = self._get_fs("emg")
                win_emg = self._ms_to_samples(self.params["window_ms"], fs_emg)
                ov_emg = self._ms_to_samples(self.params["overlap_ms"], fs_emg)
                ov_emg = min(ov_emg, max(0, win_emg - 1))  # enforce 0 <= overlap < window
                win_labels = window_labels(label, sample_size=win_emg, overlap=ov_emg)

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
                with h5py.File(os.path.join(rec_dir, "processed_emg.h5"), "w") as f:
                    if processed_emg_all is not None:
                        f.create_dataset("emg", data=processed_emg_all)
                    if win_labels is not None:
                        f.create_dataset("label", data=win_labels)

            # ---- EEG branch ----------------------------------------------------
            if getattr(self.session.config, "USE_EEG", False) and eeg_data is not None:
                np.savetxt(os.path.join(rec_dir, "raw_eeg.csv"),
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

                with h5py.File(os.path.join(rec_dir, "raw_eeg.h5"), "w") as f:
                    f.create_dataset("emg", data=eeg_data)
                    f.create_dataset("label", data=label)

                with h5py.File(os.path.join(rec_dir, "processed_eeg.h5"), "w") as f:
                    f.create_dataset("emg", data=windowed)
                    win_labels_eeg = window_labels(label, sample_size=win_eeg, overlap=ov_eeg)
                    f.create_dataset("label", data=win_labels_eeg)

        print("[processing] pipeline completed")
        if processed_emg_all is not None:
            print(f"[processing] processed EMG (all) shape: {processed_emg_all.shape}")
        if processed_emg_inlabel is not None:
            print(f"[processing] processed EMG (in-label) shape: {processed_emg_inlabel.shape}")
        return (processed_emg_all, processed_emg_inlabel)

    def closeEvent(self, event):
        """Ensure background threads and the device session are cleanly shut down on close."""
        try:
            if self._flusher is not None and self._flusher.isRunning():
                self._flusher.stop()
                self._flusher.wait(500)
            if self.session is not None:
                self.session.finish()
        except Exception:
            pass
        super().closeEvent(event)
