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
    proceed = QtCore.pyqtSignal(bool, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        title = QtWidgets.QLabel("Select Devices")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: 600;")

        self.cb_emg = QtWidgets.QCheckBox("Use EMG (Muovi)")
        self.cb_emg.setChecked(True)
        self.cb_eeg = QtWidgets.QCheckBox("Use EEG (Muovi+)")
        self.cb_eeg.setChecked(False)

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

