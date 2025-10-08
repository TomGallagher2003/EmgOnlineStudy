
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import numpy as np
import random
from sklearn.model_selection import train_test_split


from pipeline_sections.models.full_training import process_h5_files, evaluate_model, EMGDataset, CNN1D_Transformer, CNN1D, TransformerModel

seed_value = 42
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
np.random.seed(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



def load_model(model_path, model, device):
    """
    Load a saved PyTorch model from a .pth file.

    Parameters:
        model_path: Path to the .pth model file.
        model_class: The class of the model architecture.
        device: 'cuda' or 'cpu' depending on availability.

    Returns:
        model: The loaded model.
    """
    # model = model_class.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def plot_confusion_matrix(true_labels, predicted_labels, num_classes, path=None):
    """
    Plot and save a confusion matrix.

    Parameters:
        true_labels: Ground truth labels.
        predicted_labels: Predicted labels from the model.
        num_classes: Number of classes in the dataset.
    """
    conf_mat = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(10, 7))
    # num = ["0","1",       "4", "5", "8", "9", "11", "12",             "16"]
    # num = [       "2", "3",      "6", "7", "10",       "13", "14", "15", "17"]
    num = [str(i) for i in range(num_classes)]

    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=num, yticklabels=num)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    path = path if path is not None else r'C:\Users\tom03\PycharmProjects\EmgOnlineStudy\confusion_matrix_figure.png'
    plt.savefig(path)


class ChannelAdapter(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 8, k: int = 3, causal: bool = False, use_bn: bool = True):
        """
        3-tap adapter: mixes channels AND a small temporal neighborhood.
        - causal=False: symmetric padding, looks at past/now/future (default).
        - causal=True: left padding only, looks at past->now (no future leak).
        """
        super().__init__()
        assert k >= 1 and k % 2 == 1, "Use odd kernel sizes (1,3,5,...) to keep a well-defined center tap."

        if in_ch == out_ch and k == 1:
            # pure identity
            self.pad = nn.Identity()
            self.proj = nn.Identity()
            self.bn   = nn.Identity()
            self._is_identity = True
        else:
            self._is_identity = False

            if causal:
                # Conv1d doesn’t support asymmetric padding directly
                self.pad = nn.ConstantPad1d((k - 1, 0), 0.0)     # (left, right)
                conv_pad = 0
            else:
                self.pad = nn.Identity()
                conv_pad = k // 2                                # symmetric SAME-length

            self.proj = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=1, padding=conv_pad, bias=False)
            self.bn   = nn.BatchNorm1d(out_ch) if use_bn else nn.Identity()

            # init so it behaves like your old 1x1 at t0 (neighbors start at 0)
            with torch.no_grad():
                self.proj.weight.zero_()
                center = k // 2
                if in_ch % out_ch == 0:
                    g = in_ch // out_ch
                    for o in range(out_ch):
                        self.proj.weight[o, o*g:(o+1)*g, center] = 1.0 / g
                else:
                    self.proj.weight[:, :, center] = 1.0 / in_ch

    def forward(self, x):  # x: [B, in_ch, T]
        if self._is_identity:
            return x
        x = self.pad(x)
        x = self.proj(x)
        x = self.bn(x)
        return x


if __name__ == "__main__":
    for i in range(5):
        # folder_path1 = r"D:\Data\Ninapro_dataset\MIXED\hdf5_format\EB_norm_filtered"
        # folder_path1 = r"D:\Data\Ninapro_dataset\MIXED\hdf5_format\EB_norm_filtered_reduced"
        folder_path1 = r'D:\Data\Jeff_data\EMG_data\formal\mix\hdf5'
        folder_path1 = r'C:\Users\tom03\PycharmProjects\EmgOnlineStudy\data\trial_7\rec_1'

        sample_size = 512


        data1, labels1 = process_h5_files(folder_path1, sample_size=sample_size, max_zero_samples=40000, inclusion_phrase="raw_emg")


        # movement_to_remove1:list=[0,1,4,5,8,9,11,12,16]
        # movement_to_remove2:list=[cls for cls in range(18) if cls not in movement_to_remove1]
        # trainable_classes = [cls for cls in range(18) if cls not in movement_to_remove]
        # data1,labels1=remove_movement(data1, labels1, movement_to_remove1)
        data = np.concatenate([ data1], axis=0)
        labels = np.concatenate([ labels1], axis=0)
        # convert to tensor
        data = torch.tensor(data, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        indices = np.random.permutation(len(data))

        # reorder
        X_val, y_val = data[indices], labels[indices]

        print(f"Val data shape: {X_val.shape}, Val label shape: {y_val.shape}")
        # create DataLoader

        val_dataset = EMGDataset(X_val, y_val)

        val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")


        input_channels = data1.shape[2]
        length = sample_size
        embed_dim = 128
        num_heads = 8
        num_layers = 3
        num_classes = 18
        # load best model
        # model = torch.load(r"D:\Code\PhD_Code\IMECE2025_results\68_people_training\best_model_68people_99.46_5.pth")
        #model = torch.load(r"D:\Code\PhD_Code\results\TL_first_look\best_model_all_77.23_1_20251002_150824.pth")
        model = torch.load(r"C:\Users\tom03\PycharmProjects\EmgOnlineStudy\pipeline_sections\models\model.pth", weights_only=False)
        model.to(device)  # to correct device（CPU or GPU）
        _,preds, labels = evaluate_model(model, val_loader, device)
        print(preds)
        print(labels)
        report=classification_report(labels, preds, digits=4)

        # Plot confusion matrix
        plot_confusion_matrix(labels, preds, num_classes,None)

        with open(r"C:\Users\tom03\PycharmProjects\EmgOnlineStudy\classification_report.txt", "w") as f:
            f.write(report)

