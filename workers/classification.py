from typing import Optional, List, Dict, Any

import numpy as np
from PyQt5 import QtCore

from pipeline_sections.classify import run_evaluation, run_fusion_evaluation
from pipeline_sections.models.full_training import process_h5_files, evaluate_model, EMGDataset, CNN1D_Transformer, CNN1D, TransformerModel
from pipeline_sections.models.evaluation import ChannelAdapter, CNN1D, CNN1D_Transformer, TransformerModel, EMGDataset, DataLoader
from main_settings import CLASSIFY_PROCESSED_DATA, MODEL_IS_EEG

"""Post-pipeline EMG classification worker.

This module provides :class:`ClassificationWorker`, a ``QThread`` that loads the
HDF5 data written by the processing pipeline and runs a model evaluation pass.
It emits a simple decision tuple ``(pred_class, confidence)`` based on a
majority vote over predicted labels from the first evaluation run.

Overview:
    - Calls :func:`pipeline_sections.classify.run_evaluation` with the provided
      paths and hyperparameters.
    - If predictions are available, computes majority class and its proportion
      as a confidence score.
    - Emits ``finished_ok((pred_class, confidence))`` on success or
      ``failed(str)`` on error.

Notes:
    The worker assumes that the model checkpoint and folder structure are
    compatible with your pipeline's evaluation routine, and that the evaluation
    function returns a list of result dictionaries containing a ``"preds"``
    array-like entry for predicted class indices.
"""


class ClassificationWorker(QtCore.QThread):
    """Runs post-pipeline classification using H5 written during pipeline.

    Signals:
        finished_ok (object): Emitted with a tuple ``(pred_class, confidence)``.
            ``pred_class`` is a stringified integer label; ``confidence`` is the
            proportion of votes for that class in ``[0.0, 1.0]``. If no results
            are available, the worker emits ``("", 0.0)``.
        failed (str): Emitted with a human-readable error message if evaluation
            fails for any reason.

    Typical usage:
        >>> worker = ClassificationWorker(
        ...     folder_path="/path/to/h5",
        ...     model_path="/path/to/model.pt",
        ...     report_save_path="/path/to/report_dir",
        ...     sample_size=512,
        ...     batch_size=512,
        ...     num_classes=30,
        ...     repeats=1,
        ...     parent=self,
        ... )
        >>> worker.finished_ok.connect(self.on_prediction_ready)
        >>> worker.failed.connect(self.on_prediction_failed)
        >>> worker.start()
    """

    finished_ok = QtCore.pyqtSignal(object)  # emits (A, B)
    failed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        folder_path: str,
        model_path: str,
        report_save_path: str,
        *,
        sample_size: int = 512,
        batch_size: int = 512,
        num_classes: int = 30,
        repeats: int = 1,
        parent=None,
    ):
        """Initialize the classification worker.

        Args:
            folder_path: Directory containing H5 files produced by the pipeline.
            model_path: Path to the trained model checkpoint or pickle.
            report_save_path: Directory where evaluation artifacts/reports are saved.
            sample_size: Window length (samples) used during evaluation inference.
            batch_size: Batch size for the evaluation dataloader.
            num_classes: Number of distinct class labels expected by the model.
            repeats: Number of evaluation runs to perform (e.g., for ensembling).
            parent: Optional QObject parent for Qt ownership semantics.
        """
        super().__init__(parent)
        self.folder_path = folder_path
        self.model_path = model_path
        self.report_save_path = report_save_path
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.repeats = repeats

    def run(self) -> None:
        """Execute the evaluation and emit a majority-vote prediction.

        Workflow:
            1. Invoke :func:`run_evaluation` with the configured parameters.
            2. If no results are returned, emit ``("", 0.0)``.
            3. From the first run's result, read ``preds`` and compute the
               majority label and its relative frequency as confidence.
            4. Emit ``finished_ok((pred_class, confidence))`` on success.

        On any exception, a descriptive error string is emitted via ``failed``.
        """
        try:
            # Call your evaluation function once (or repeats times if you prefer)
            results: List[Dict[str, Any]] = run_evaluation(
                folder_path=self.folder_path,
                model_path=self.model_path,
                report_save_path=self.report_save_path,
                sample_size=self.sample_size,
                batch_size=self.batch_size,
                num_classes=self.num_classes,
                repeats=self.repeats,
                use_full_model_pickle=True,
                device=None,  # auto-selects CUDA/CPU inside
                confusion_fixed_name=None,
                use_processed=CLASSIFY_PROCESSED_DATA,
                model_is_eeg=MODEL_IS_EEG
            )

            if not results:
                self.finished_ok.emit(("", 0.0))
                return

            # Take the first run's predictions and compute a simple majority vote
            first = results[0]
            preds = np.asarray(first.get("preds", []))
            print("predicitions->", preds)
            if preds.size == 0:
                self.finished_ok.emit(("", 0.0))
                return

            vals, counts = np.unique(preds, return_counts=True)
            idx = int(np.argmax(counts))
            pred_class = str(int(vals[idx]))
            confidence = float(counts[idx]) / float(preds.size)  # proportion as confidence

            self.finished_ok.emit((pred_class, confidence))

        except Exception as e:
            self.failed.emit(str(e))


class FusionClassificationWorker(QtCore.QThread):
    """Runs post-pipeline fusion classification using both EMG and EEG H5 files.

    Signals:
        finished_ok (object): Emitted with ``(pred_class, confidence)``.
        failed (str): Emitted on error.
    """

    finished_ok = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        folder_path: str,
        model_path: str,
        report_save_path: str,
        *,
        sample_size: int = 512,
        batch_size: int = 512,
        num_classes: int = 30,
        parent=None,
    ):
        super().__init__(parent)
        self.folder_path = folder_path
        self.model_path = model_path
        self.report_save_path = report_save_path
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.num_classes = num_classes

    def run(self) -> None:
        try:
            result = run_fusion_evaluation(
                folder_path=self.folder_path,
                model_path=self.model_path,
                report_save_path=self.report_save_path,
                sample_size=self.sample_size,
                batch_size=self.batch_size,
                num_classes=self.num_classes,
                device=None,
            )
            preds = np.asarray(result.get("preds", []))
            if preds.size == 0:
                self.finished_ok.emit(("", 0.0))
                return
            vals, counts = np.unique(preds, return_counts=True)
            idx = int(np.argmax(counts))
            pred_class = str(int(vals[idx]))
            confidence = float(counts[idx]) / float(preds.size)
            self.finished_ok.emit((pred_class, confidence))
        except Exception as e:
            self.failed.emit(str(e))
