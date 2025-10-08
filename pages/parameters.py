from PyQt5 import QtCore, QtGui, QtWidgets

from emg_default_settings import (DEFAULT_WINDOW_MS, DEFAULT_LOWER_HZ, DEFAULT_UPPER_HZ,
                                  DEFAULT_RECORDING_LENGTH_S, DEFAULT_OVERLAP_MS, DEFAULT_NOTCH_HZ,
                                  DEFAULT_USE_AUTO_SEGMENTATION, DEFAULT_APPLY_NORMALISATION)


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

        # --- Minimal setters to allow programmatic defaults ---
        def set_type(self, t: str):
            if t in ParametersPage.FILTER_OPTIONS:
                self.type_box.setCurrentText(t)
                self._update_rows()

        def set_pass(self, lower: float = None, upper: float = None):
            self.set_type("Pass")
            self.lower_edit.setText("" if lower is None else str(lower))
            self.upper_edit.setText("" if upper is None else str(upper))

        def set_notch(self, center: float = None):
            self.set_type("Notch")
            self.center_edit.setText("" if center is None else str(center))

        def set_none(self):
            self.set_type("None")
            # Clear any lingering values
            self.lower_edit.clear()
            self.upper_edit.clear()
            self.center_edit.clear()

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
        self.overlap_ms_edit.setPlaceholderText("")
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

        # --- New: Apply EMG Defaults button (visible only if EMG is in use) ---
        self.btn_apply_emg_defaults = QtWidgets.QPushButton("Apply EMG Defaults")
        self.btn_apply_emg_defaults.setToolTip(
            "Set overlap, EMG filters (Pass + Notch), auto-segmentation, and recording length to defaults."
        )
        self.btn_apply_emg_defaults.clicked.connect(self._apply_emg_defaults)
        # put button aligned right in the EMG group
        emg_btn_row = QtWidgets.QHBoxLayout()
        emg_btn_row.addStretch(1)
        emg_btn_row.addWidget(self.btn_apply_emg_defaults)
        emg_lay.addRow(emg_btn_row)

        self.emg_group.setVisible(self.use_emg)
        self.btn_apply_emg_defaults.setVisible(self.use_emg)

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

    def _apply_emg_defaults(self):
        """
        Apply EMG defaults:
          - Overlap (ms)       -> DEFAULT_OVERLAP_MS
          - EMG First Filter   -> Pass [DEFAULT_LOWER_HZ, DEFAULT_UPPER_HZ]
          - EMG Second Filter  -> Notch @ DEFAULT_NOTCH_HZ
          - EMG Third Filter   -> None
          - Auto segmentation  -> DEFAULT_USE_AUTO_SEGMENTATION
          - Recording length   -> DEFAULT_RECORDING_LENGTH_S
        """
        # Overlap
        self.overlap_ms_edit.setText(str(DEFAULT_OVERLAP_MS))

        # Filters
        self.emg_a.set_pass(lower=DEFAULT_LOWER_HZ, upper=DEFAULT_UPPER_HZ)
        self.emg_b.set_notch(center=DEFAULT_NOTCH_HZ)
        self.emg_c.set_none()

        # Auto-segmentation
        self.emg_auto_seg.setChecked(DEFAULT_USE_AUTO_SEGMENTATION)
        self.use_normalisation.setChecked(DEFAULT_APPLY_NORMALISATION)

        # Recording length
        self.length_edit.setText(str(DEFAULT_RECORDING_LENGTH_S))

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
