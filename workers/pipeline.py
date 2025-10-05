from PyQt5 import QtCore


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

