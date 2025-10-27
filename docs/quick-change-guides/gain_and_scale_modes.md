---
title: Device Config & Changing Probe Device Modes
summary: Change EMG/EEG voltage scaling ratios when using different amplifier gains.
---

# Gain & Scaling Modes

If you want to switch hardware modes for different gains or test modes for debugging. This guides explains where to find these settings.
## Where
- **File:** `config.py` 
- **Fields:** `self.EMG_MODE` & `self.EEG_MODE` 
- **Also see:** `processing.py` for counts→voltage scaling

## How

Update these fields based on the allowed values as described in the file (also described in the device communication protocol documents)

I would not recommend touching anything else in this file, as the majority of it represents the initial hard-coded constants provided by OTB. Things like IP and ports are defined here, but they should never change. 