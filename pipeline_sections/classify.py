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
):
    """
    Callable entrypoint that mirrors existing evaluation logic.

    Args:
        folder_path: path to a folder containing HDF5 raw emg
        model_path:  path to .pth model file
        sample_size, batch_size, num_classes: same meanings as in your script
        repeats: how many times to run (like your for i in range(5))
        use_full_model_pickle: True if the checkpoint is a pickled full model
        device: pass a torch.device, otherwise auto-selects cuda/cpu
        report_save_path: where to write the classification report
        confusion_fixed_name: if given, reuse a fixed filename for the confusion matrix;
                              otherwise  original plot function decides (or you can keep i-based)
    Returns:
        A list of dicts with {"report": str, "preds": np.ndarray, "labels": np.ndarray}
        (one entry per repeat)
    """
    folder_path = Path(folder_path)
    model_path = Path(model_path)
    out_dir = Path(report_save_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # load data (one folder)
    data1, labels1 = process_h5_files(str(folder_path), sample_size=sample_size, max_zero_samples=400, inclusion_phrase="raw_emg")

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