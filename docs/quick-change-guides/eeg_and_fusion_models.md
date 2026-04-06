---
title: EEG & Fusion Models
summary: Use an EEG-only model or a combined EMG+EEG fusion model for classification
---

# EEG & Fusion Models

This guide covers two scenarios:

- **EEG-only** — classify using EEG data alone
- **Fusion** — classify using both EMG and EEG together

Both are already wired into the program. You just need to train the relevant model, place the file in the right folder, and set one variable.

---

## How to switch modes

Open `main_settings.py` and set `MODEL_MODE` to one of three options:

```python
# main_settings.py

MODEL_PATH        = "./pipeline_sections/models/model.pth"        # EMG model
EEG_MODEL_PATH    = "./pipeline_sections/models/eeg_model.pth"    # EEG model
FUSION_MODEL_PATH = "./pipeline_sections/models/fusion_model.pth" # Fusion model

# Set ONE of: "emg" | "eeg" | "fusion"
MODEL_MODE = "emg"
```

| `MODEL_MODE` | Data used |
|---|---|
| `"emg"` | EMG only |
| `"eeg"` | EEG only |
| `"fusion"` | EMG + EEG together |

!!! note "Device requirements"
    EEG and Fusion modes require **EEG** to be selected on the Device Selection page. Fusion additionally requires **EMG** to be selected.

---

## Scenario A — EEG-only model

### 1. Train your EEG model

Your model must accept EEG windows of shape `(B, eeg_channels, T)` — the same shape convention as the existing EMG models, just with EEG channel count instead.

Any of the existing architectures work. Example:

```python
from pipeline_sections.models.full_training import CNN1D_Transformer

model = CNN1D_Transformer(
    input_channels=64,   # number of EEG channels
    length=512,          # window length in samples
    embed_dim=128,
    num_heads=8,
    num_layers=3,
    num_classes=30,
)
```

Save the full model (not just the weights):

```python
torch.save(model, "eeg_model.pth")
```

### 2. Place the file

Copy `eeg_model.pth` to:

```
pipeline_sections/models/eeg_model.pth
```

### 3. Set the mode

```python
# main_settings.py
EEG_MODEL_PATH = "./pipeline_sections/models/eeg_model.pth"
MODEL_MODE     = "eeg"
```

### 4. Run

Start the app, select **EEG** on the Device Selection page, and proceed as normal. The classification result will appear in the Classification Results panel after each recording.

---

## Scenario B — Fusion model (EMG + EEG)

The fusion model receives two inputs — one window of EMG data and one window of EEG data — and outputs a single classification. How those two inputs are combined internally is up to whoever builds the model.

The only requirement the program has is that the model accepts two inputs in this order:

```python
logits = model(emg_data, eeg_data)
```

Where:
- `emg_data` has shape `(B, emg_channels, T)`
- `eeg_data` has shape `(B, eeg_channels, T)`

A ready-to-use `FusionModel` is available in `pipeline_sections/models/full_training.py` as a starting point. You can also implement a completely different architecture — as long as it takes those two inputs and returns class scores, it will work.

### 1. Train your fusion model

Example using the built-in architecture:

```python
from pipeline_sections.models.full_training import FusionModel

model = FusionModel(
    emg_channels=32,
    eeg_channels=64,
    length=512,
    embed_dim=128,
    num_heads=8,
    num_layers=3,
    num_classes=30,
)
```

Your training data needs paired EMG and EEG windows with the same labels. A simple dataset class:

```python
from torch.utils.data import Dataset, DataLoader

class PairedDataset(Dataset):
    def __init__(self, emg, eeg, labels):
        self.emg, self.eeg, self.labels = emg, eeg, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.emg[idx], self.eeg[idx], self.labels[idx]

dataset = PairedDataset(emg_windows, eeg_windows, labels)
loader  = DataLoader(dataset, batch_size=512, shuffle=True)
```

Training loop:

```python
for emg_batch, eeg_batch, labels in train_loader:
    emg_batch = emg_batch.permute(0, 2, 1).to(device)
    eeg_batch = eeg_batch.permute(0, 2, 1).to(device)
    labels    = labels.to(device)

    optimizer.zero_grad()
    out  = model(emg_batch, eeg_batch)
    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()
```

### 2. Save and place the file

```python
torch.save(model, "fusion_model.pth")
```

Copy it to:

```
pipeline_sections/models/fusion_model.pth
```

### 3. Set the mode

```python
# main_settings.py
FUSION_MODEL_PATH = "./pipeline_sections/models/fusion_model.pth"
MODEL_MODE        = "fusion"
```

### 4. Run

Select **both EMG and EEG** on the Device Selection page. After each recording the program will classify using both signals and show the result as normal.

---

## Window sizes (fusion)

EMG and EEG are recorded at different sample rates, so the same time window produces a different number of samples for each:

| Modality | Sample rate | 256 ms window |
|---|---|---|
| EMG | 2000 Hz | 512 samples |
| EEG (downsampled) | 500 Hz | 128 samples |

Make sure your fusion model is built to handle these different lengths.

!!! warning "Window count mismatch"
    If EMG and EEG produce a different number of windows, the program automatically trims to the shorter count. To avoid this, use the same overlap setting for both modalities.

---

## Switching back to EMG-only

```python
# main_settings.py
MODEL_MODE = "emg"
```

---

## Checklist

- [ ] Model saved with `torch.save(model, ...)` — the full model object, not just weights
- [ ] Model file placed in `pipeline_sections/models/`
- [ ] `MODEL_MODE` set correctly in `main_settings.py`
- [ ] Correct devices selected in the UI
- [ ] For fusion: model `forward` method accepts `(emg, eeg)` as two separate inputs
