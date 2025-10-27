from typing import Dict, Any

from PyQt5 import QtWidgets

from pages.device_select import DeviceSelectPage
from pages.experiment import ExperimentPage
from pages.parameters import ParametersPage

"""Top-level application window and page flow.

This module defines :class:`MainWindow`, a simple stacked-page controller for
the online study GUI. The flow is:

    DeviceSelect → Parameters → Experiment

The window owns a :class:`QtWidgets.QStackedWidget` and swaps in the pages as
the user progresses. No behavioral changes—only documentation for mkdocstrings.
"""


class MainWindow(QtWidgets.QMainWindow):
    """Main window hosting the multi-page online study workflow.

    The window presents three sequential pages backed by a stacked widget:

    1) :class:`pages.device_select.DeviceSelectPage` — choose EMG/EEG devices.
    2) :class:`pages.parameters.ParametersPage` — set recording and processing options.
    3) :class:`pages.experiment.ExperimentPage` — run capture, pipeline, and classification.

    Notes:
        The page transitions are driven by Qt signals emitted from each page.
        Parameters selected on the Parameters page are forwarded to the
        Experiment page upon navigation.
    """

    def __init__(self):
        """Initialize the window, stack, and the first page (Device Select)."""
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
    def _go_params(self, use_emg: bool, use_eeg: bool) -> None:
        """Transition from Device Select to Parameters page.

        Args:
            use_emg: Whether EMG is enabled, as chosen on the Device Select page.
            use_eeg: Whether EEG is enabled, as chosen on the Device Select page.
        """
        self.use_emg, self.use_eeg = use_emg, use_eeg
        self.page_params = ParametersPage(use_emg=self.use_emg, use_eeg=self.use_eeg)
        self.page_params.proceed.connect(self._go_experiment)
        self.page_params.btn_back.clicked.connect(lambda: self.stack.setCurrentWidget(self.page_select))
        self.stack.addWidget(self.page_params)
        self.stack.setCurrentWidget(self.page_params)

    def _go_experiment(self, params: dict) -> None:
        """Transition from Parameters to Experiment page.

        Args:
            params: Dictionary of user-selected parameters to pass through to the
                :class:`pages.experiment.ExperimentPage`.
        """
        self.params = params
        self.page_experiment = ExperimentPage(self.use_emg, self.use_eeg, params=self.params)
        self.stack.addWidget(self.page_experiment)
        self.stack.setCurrentWidget(self.page_experiment)

    def closeEvent(self, event) -> None:
        """Ensure child pages clean up resources before the window closes.

        Calls ``page_experiment.close()`` if present so background threads
        (e.g., flusher) and device sessions are shut down cleanly.
        """
        try:
            if hasattr(self, "page_experiment"):
                # ensure flusher stops and session closes
                self.page_experiment.close()
        except Exception:
            pass
        super().closeEvent(event)
