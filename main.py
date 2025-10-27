# main.py
# PyQt5 app: device select -> parameter entry -> device check/init -> random movement -> arc timer
# -> run_pipeline() -> (classification if EMG: in-label only)
# - Device check/init happens right after parameters screen (session kept open)
# - Recording runs in background; pipeline runs after; classification runs only if EMG present

"""Launcher for the movement-timer desktop app.

This module boots the PyQt5 GUI that drives your end-to-end EMG/EEG workflow:
device selection → parameter entry → device init/check → randomized movement
prompting with an arc timer → post-recording pipeline (feature extraction /
training / evaluation) → optional in-label classification for EMG.

Usage:
    Run this file directly to start the application:

    ```bash
    python -m movement_timer
    # or
    python main.py
    ```

Architecture:
    - The top-level window is provided by :class:`pages.main.MainWindow`.
    - Recording is performed in the background once parameters are confirmed.
    - The processing/training/evaluation pipeline executes after recording.
    - If EMG channels are present, an in-label classification pass runs.

Notes:
    The module imports multiple modeling symbols from two different subpackages
    (``pipeline_sections.models.full_training`` and
    ``pipeline_sections.models.evaluation``). This is intentional to make those
    classes available to the GUI and any late binding performed inside
    :class:`pages.main.MainWindow`. Do not remove or reorder these imports,
    because some code paths may rely on them being import-time side effects.

"""

import sys

from PyQt5 import QtWidgets

from pages.main import MainWindow
from pipeline_sections.models.full_training import (
    process_h5_files,
    evaluate_model,
    EMGDataset,
    CNN1D_Transformer,
    CNN1D,
    TransformerModel,
)
from pipeline_sections.models.evaluation import (
    ChannelAdapter,
    CNN1D,
    CNN1D_Transformer,
    TransformerModel,
    EMGDataset,
    DataLoader,
)


# ------------------------------------------------------------------------------


def main() -> None:
    """Start the PyQt5 event loop and show the main window.

    This function constructs a :class:`QtWidgets.QApplication`, instantiates the
    :class:`pages.main.MainWindow`, and enters the Qt event loop. It does not
    return until the application exits.

    Side Effects:
        Initializes the global Qt application instance and blocks in the event
        loop until the user closes the window.

    Raises:
        SystemExit: Propagated when the Qt event loop finishes and the process
            exits with the return code from ``app.exec_()``.
    """
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
