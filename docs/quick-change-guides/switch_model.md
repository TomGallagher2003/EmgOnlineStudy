---
title: Switching Models
summary: Switch models and update classification input to fit
---

# Switching Models

Models can be switched simply in `main_settings.py`. This configuration file has 3 important fields to define the model.

```python
# Model Info
MODEL_PATH = str("." / "pipeline_sections" / "models" / "model.pth")
MODEL_IS_EEG = False             # False: EMG, True: EEG
CLASSIFY_PROCESSED_DATA = False  # False: raw data used for classification, True: processed data used for classification

```

All of these fields must be correct to match the models expected input format, and determine which data will be used for classification.