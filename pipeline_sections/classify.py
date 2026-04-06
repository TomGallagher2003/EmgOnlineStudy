# evaluation_api.py
import os
from pathlib import Path
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from pipeline_sections.models.full_training import process_h5_files, evaluate_model, EMGDataset, CNN1D_Transformer, CNN1D, TransformerModel
from pipeline_sections.models.evaluation import ChannelAdapter, CNN1D, CNN1D_Transformer, TransformerModel, EMGDataset, DataLoader


from pipeline_sections.models.evaluation import (
    process_h5_files,
    evaluate_model,
    EMGDataset,
    CNN1D_Transformer,
    CNN1D,
    TransformerModel,
    plot_confusion_matrix,
    ChannelAdapter
)

class _FusionDataset(torch.utils.data.Dataset):
    def __init__(self, emg_data, eeg_data, labels):
        self.emg = emg_data
        self.eeg = eeg_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.emg[idx], dtype=torch.float32),
            torch.tensor(self.eeg[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )

"""Lightweight evaluation wrapper for EMG classification models.

This module provides :func:`run_evaluation`, a callable entry point that mirrors
your existing notebook/script logic so the UI (or other tools) can trigger a
repeatable evaluation pass on a folder of HDF5 artifacts.

Workflow overview:
    1) Load EMG windows and labels from a single folder via ``process_h5_files``.
    2) Build a ``DataLoader`` for validation (shuffled each repeat).
    3) Load a model checkpoint (either a full pickled model or state_dict route).
    4) Run inference with ``evaluate_model`` to obtain predictions and labels.
    5) Produce a ``classification_report`` and confusion matrix image.
    6) Return a list of result dicts, one per repeat.

Reproducibility:
    Global seeds are set (NumPy / PyTorch CPU & CUDA) and cuDNN is forced into
    deterministic mode with benchmarking disabled to reduce run-to-run variance.
"""

# keep the same reproducibility settings/style
seed_value = 42
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
np.random.seed(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def run_evaluation(
    folder_path: str,
    model_path: str,
    report_save_path: str,
    *,
    sample_size: int = 512,
    batch_size: int = 512,
    num_classes: int = 30,
    repeats: int = 1,
    use_full_model_pickle: bool = True,
    device: torch.device | None = None,
    confusion_fixed_name: str | None = None,
    use_processed: bool = False,
    model_is_eeg: bool = False
):
    """Evaluate a trained model on EMG windows stored in a folder.

    This callable mirrors your existing evaluation script so it can be invoked
    programmatically (e.g., from a GUI worker). It shuffles the validation set
    on each repeat, computes predictions, writes a classification report, plots
    a confusion matrix, and returns key artifacts.

    Args:
        folder_path:
            Path to a folder containing HDF5 files with EMG windows/labels
            (as produced by your pipeline). The helper ``process_h5_files`` is
            used to load a single folder.
        model_path:
            Path to the model checkpoint. If ``use_full_model_pickle`` is True,
            this should be a pickled full model; otherwise the function will
            instantiate a ``CNN1D_Transformer`` and load a state_dict.
        report_save_path:
            File path to write the ``classification_report`` text output.
        sample_size:
            Window length (samples) used when loading/validating.
        batch_size:
            Batch size for the ``DataLoader``.
        num_classes:
            Number of classes the model predicts.
        repeats:
            How many evaluation repeats to perform (reshuffles each time).
        use_full_model_pickle:
            Whether to ``torch.load`` a full serialized model object.
        device:
            Optional ``torch.device``. If ``None``, chooses CUDA when available.
        confusion_fixed_name:
            If provided, indicates a fixed name strategy for the confusion
            matrix. (Current implementation saves to ``confusion_matrix.png``.)\
        use_processed:
            boolean indicating whether to search for processed or raw data files

    Returns:
        list[dict]:
            A list of dictionaries (one per repeat) with keys:
            - ``"report"`` (str): the classification report text.
            - ``"preds"`` (np.ndarray): predicted class indices.
            - ``"labels"`` (np.ndarray): ground-truth class indices.

    Notes:
        - ``process_h5_files`` is called with ``inclusion_phrase="raw_emg"`` and
          ``max_zero_samples=400`` to mirror prior behavior.
        - The confusion matrix is currently saved to ``<folder_path>/confusion_matrix.png``.
          If you later vary naming by repeat, use ``confusion_fixed_name``.
    """
    folder_path = Path(folder_path)
    model_path = Path(model_path)
    out_dir = Path(report_save_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    inclusion_p1 = "processed_" if use_processed else "raw_"
    inclusion_p2 = "eeg" if model_is_eeg else "emg"
    inclusion_phrase = inclusion_p1 + inclusion_p2

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # load data (one folder)
    data1, labels1 = process_h5_files(str(folder_path), sample_size=sample_size, max_zero_samples=400, inclusion_phrase=inclusion_phrase)

    # tensors
    data = torch.tensor(data1, dtype=torch.float32)
    labels = torch.tensor(labels1, dtype=torch.long)

    # build DataLoader each repeat (to reshuffle)
    results = []
    for i in range(repeats):
        indices = np.random.permutation(len(data))
        X_val, y_val = data[indices], labels[indices]
        print(f"Val data shape: {X_val.shape}, Val label shape: {y_val.shape}")

        val_dataset = EMGDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # load model (keep your behavior)
        if use_full_model_pickle:
            model = torch.load(str(model_path), map_location=device, weights_only=False)
        else:
            # If you switch to state_dict in future, instantiate and load here.
            model = CNN1D_Transformer(in_channels=data1.shape[2], num_classes=num_classes)
            state = torch.load(str(model_path), map_location=device)
            model.load_state_dict(state)

        model.to(device)  # to correct device（CPU or GPU）
        # evaluate
        _, preds, labels_np = evaluate_model(model, val_loader, device)

        # report
        report = classification_report(labels_np, preds, digits=4, zero_division=0)

        if confusion_fixed_name is not None:
            plot_confusion_matrix(labels_np, preds, num_classes, path=os.path.join(folder_path, "confusion_matrix.png"))
        else:
            plot_confusion_matrix(labels_np, preds, num_classes, path=os.path.join(folder_path, "confusion_matrix.png"))

        # write report
        with open(report_save_path, "w", encoding="utf-8") as f:
            f.write(report)

        results.append({
            "report": report,
            "preds": np.asarray(preds),
            "labels": np.asarray(labels_np),
        })

    return results


def run_fusion_evaluation(
    folder_path: str,
    model_path: str,
    report_save_path: str,
    *,
    sample_size: int = 512,
    batch_size: int = 512,
    num_classes: int = 30,
    device: torch.device | None = None,
):
    """Evaluate a FusionModel on paired EMG + EEG H5 files in folder_path.

    Loads ``raw_emg.h5`` and ``raw_eeg.h5`` from the same folder, aligns their
    window counts (truncates to the shorter), runs inference through the fusion
    model, and writes a classification report.

    Args:
        folder_path: Directory containing raw_emg.h5 and raw_eeg.h5.
        model_path: Path to a pickled FusionModel checkpoint.
        report_save_path: File path for the classification_report text output.
        sample_size: Window length used for both modalities.
        batch_size: DataLoader batch size.
        num_classes: Number of output classes.
        device: Optional torch.device; auto-selects CUDA when None.

    Returns:
        dict with keys ``"report"``, ``"preds"``, ``"labels"``.
    """
    folder_path = Path(folder_path)
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    emg_data, emg_labels = process_h5_files(str(folder_path), sample_size=sample_size,
                                             max_zero_samples=400, inclusion_phrase="raw_emg")
    eeg_data, eeg_labels = process_h5_files(str(folder_path), sample_size=sample_size,
                                             max_zero_samples=400, inclusion_phrase="raw_eeg")

    n = min(len(emg_labels), len(eeg_labels))
    emg_data, eeg_data, labels = emg_data[:n], eeg_data[:n], emg_labels[:n]

    indices = np.random.permutation(n)
    emg_data, eeg_data, labels = emg_data[indices], eeg_data[indices], labels[indices]

    dataset = _FusionDataset(emg_data, eeg_data, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = torch.load(str(model_path), map_location=device, weights_only=False)
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for emg_b, eeg_b, lbl_b in loader:
            emg_b = emg_b.permute(0, 2, 1).to(device)
            eeg_b = eeg_b.permute(0, 2, 1).to(device)
            lbl_b = lbl_b.to(device)
            out = model(emg_b, eeg_b)
            _, predicted = torch.max(out, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(lbl_b.cpu().numpy())

    from sklearn.metrics import classification_report as skl_report
    report = skl_report(all_labels, all_preds, digits=4, zero_division=0)
    Path(report_save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_save_path, "w", encoding="utf-8") as f:
        f.write(report)

    plot_confusion_matrix(all_labels, all_preds, num_classes,
                          path=str(folder_path / "confusion_matrix.png"))

    return {"report": report, "preds": np.asarray(all_preds), "labels": np.asarray(all_labels)}


if __name__ == "__main__":
    # test
    _ = run_evaluation(
        folder_path=r'C:\Users\tom03\PycharmProjects\EmgOnlineStudy\data\trial_1\rec_1',
        model_path=r"C:\Users\tom03\PycharmProjects\EmgOnlineStudy\pipeline_sections\models\model.pth",
        sample_size=512,
        batch_size=512,
        num_classes=30,
        repeats=5,
        use_full_model_pickle=True,
        report_save_path=r"C:\Users\tom03\PycharmProjects\EmgOnlineStudy\data\classification_outputs\classification_report.txt",
        confusion_fixed_name=None,
    )
