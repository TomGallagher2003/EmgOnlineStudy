from typing import Optional, List, Dict, Any

import numpy as np
from PyQt5 import QtCore

from pipeline_sections.classify import run_evaluation
from pipeline_sections.models.full_training import process_h5_files, evaluate_model, EMGDataset, CNN1D_Transformer, CNN1D, TransformerModel
from pipeline_sections.models.evaluation import ChannelAdapter, CNN1D, CNN1D_Transformer, TransformerModel, EMGDataset, DataLoader



class ClassificationWorker(QtCore.QThread):
    """Runs post-pipeline classification using H5 written during pipeline."""
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
        super().__init__(parent)
        self.folder_path = folder_path
        self.model_path = model_path
        self.report_save_path = report_save_path
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.repeats = repeats

    def run(self):
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
