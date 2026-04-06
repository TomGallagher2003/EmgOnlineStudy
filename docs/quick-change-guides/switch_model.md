---
title: Switching Models
summary: Switch models and update classification input to fit
---

# Switching Models

Models are configured in `main_settings.py`.

```python
# Model Info
MODEL_PATH        = "./pipeline_sections/models/model.pth"        # EMG model
EEG_MODEL_PATH    = "./pipeline_sections/models/eeg_model.pth"    # EEG model
FUSION_MODEL_PATH = "./pipeline_sections/models/fusion_model.pth" # Fusion model

# Set ONE of: "emg" | "eeg" | "fusion"
MODEL_MODE = "emg"

CLASSIFY_PROCESSED_DATA = False  # False: raw data, True: processed (windowed) data
```

Set `MODEL_MODE` to select which modality is classified, and point the matching `*_MODEL_PATH` to your checkpoint. All fields must be consistent with the model's expected input.

For EEG-only or fusion workflows, see the [EEG & Fusion Models](eeg_and_fusion_models.md) guide.