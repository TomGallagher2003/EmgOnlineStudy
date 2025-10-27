from PyPy5 import QtCore


class PipelineWorker(QtCore.QThread):
    """Background worker to run the data-processing pipeline.

    This QThread wrapper executes a user-provided pipeline function off the UI
    thread and reports either a successful result or a failure via signals.

    The pipeline function should accept ``(movement_name, data)`` and return a
    tuple ``(processed_emg_all, processed_emg_inlabel)`` suitable for downstream
    visualisation or model training.

    Signals:
        finished_ok (object): Emitted with the pipeline output tuple
            ``(processed_emg_all, processed_emg_inlabel)`` when processing
            completes successfully.
        failed (str): Emitted with a human-readable error message if processing
            raises an exception.

    Notes:
        - This worker does not mutate UI state directly; connect to its signals
          to update widgets safely on the main thread.
        - The input ``data`` should already be validated/structured by the
          recording step (e.g., NumPy arrays, dict of arrays, etc.).
    """

    finished_ok = QtCore.pyqtSignal(object)  # emits (processed_emg_all, processed_emg_inlabel)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, pipeline_fn, movement_name, data, parent=None):
        """Initialize the worker.

        Args:
            pipeline_fn: Callable accepting ``(movement_name, data)`` and
                returning ``(processed_emg_all, processed_emg_inlabel)``.
            movement_name: String label for the movement being processed; passed
                through to ``pipeline_fn`` (often used for routing/metadata).
            data: Raw or pre-structured recording payload consumed by
                ``pipeline_fn``.
            parent: Optional QObject parent for standard Qt ownership semantics.
        """
        super().__init__(parent)
        self.pipeline_fn = pipeline_fn
        self.movement_name = movement_name
        self.data = data

    def run(self):
        """Execute the pipeline on a background thread.

        Calls ``pipeline_fn(movement_name, data)`` and emits ``finished_ok`` with
        the returned tuple. If any exception occurs, emits ``failed`` with the
        error string and returns.
        """
        try:
            processed_tuple = self.pipeline_fn(self.movement_name, self.data)
            self.finished_ok.emit(processed_tuple)
        except Exception as e:
            self.failed.emit(str(e))
