"""Device selection page (GUI).

This module defines :class:`DeviceSelectPage`, a minimal PyQt5 widget that lets
the user choose whether to use EMG (Muovi) and/or EEG (Muovi+) for the session.
On confirmation, it emits a ``proceed(use_emg: bool, use_eeg: bool)`` signal to
advance the app flow.

Notes:
    - The extra imports are intentionally preserved for project-wide consistency
      and mkdocstrings discovery. Do not remove or reorder imports.
"""

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from pipeline_sections.filters import selective_filter
from pipeline_sections.reduce_eeg_samples import reduce_eeg
from util.images import Images
from util.recording import Session
from util.movement_segmentation import detect_movement_mask
from pipeline_sections.normalisation import normalise_data
from pipeline_sections.windows import window_data


class DeviceSelectPage(QtWidgets.QWidget):
    """Simple device picker for EMG/EEG.

    Signals:
        proceed (bool, bool): Emitted with the states of the EMG and EEG
            checkboxes respectively, to indicate which modalities to enable.
    """

    proceed = QtCore.pyqtSignal(bool, bool)

    def __init__(self, parent=None):
        """Build the device-selection UI.

        Args:
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)

        # --- Styles: larger fonts + bigger checkbox indicator
        self.setStyleSheet("""
            QLabel#title { font-size: 24px; font-weight: 700; }
            QCheckBox { font-size: 18px; }
            QCheckBox::indicator { width: 22px; height: 22px; }
            QPushButton { font-size: 16px; padding: 8px 18px; }
        """)

        # Title
        title = QtWidgets.QLabel("Select Devices", self)
        title.setObjectName("title")
        title.setAlignment(QtCore.Qt.AlignCenter)

        # Checkboxes
        self.cb_emg = QtWidgets.QCheckBox("Use EMG (Muovi)", self)
        self.cb_emg.setChecked(True)
        self.cb_eeg = QtWidgets.QCheckBox("Use EEG (Muovi+)", self)
        self.cb_eeg.setChecked(False)

        # Continue button
        btn_continue = QtWidgets.QPushButton("Continue", self)
        btn_continue.setFixedHeight(40)
        btn_continue.clicked.connect(self._on_continue)

        # --- Center everything horizontally & vertically ---
        # Outer layout fills the widget; inner layout is centered.
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 24)

        inner = QtWidgets.QVBoxLayout()
        inner.setSpacing(16)
        inner.setAlignment(QtCore.Qt.AlignCenter)  # centers children horizontally & vertically as a block

        # Add widgets to inner (centered) stack
        inner.addWidget(title, 0, QtCore.Qt.AlignHCenter)
        inner.addSpacing(8)
        inner.addWidget(self.cb_emg, 0, QtCore.Qt.AlignHCenter)
        inner.addWidget(self.cb_eeg, 0, QtCore.Qt.AlignHCenter)
        inner.addSpacing(12)
        inner.addWidget(btn_continue, 0, QtCore.Qt.AlignHCenter)

        # Use stretch to keep the inner block vertically centered within the page
        outer.addStretch(1)
        outer.addLayout(inner)
        outer.addStretch(1)

    def _on_continue(self):
        """Emit the chosen modality configuration and let the caller advance."""
        self.proceed.emit(self.cb_emg.isChecked(), self.cb_eeg.isChecked())
