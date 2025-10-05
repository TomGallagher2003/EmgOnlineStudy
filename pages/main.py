from typing import Dict, Any

from PyQt5 import QtWidgets

from pages.device_select import DeviceSelectPage
from pages.experiment import ExperimentPage
from pages.parameters import ParametersPage


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