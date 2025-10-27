---
title: Plotting Guide (view_csv.py)
summary: Plot a single channel, set y-limits, or choose a channel list in seconds.
---

# Plotting Quick Tweaks

Rapid tips to adjust plotting without digging into code. These examples assume you’re using `view_csv.py` to inspect CSV/HDF5 data.

## One channel view
```python
# view_csv.py (example toggles)
SINGLE_CHANNEL_MODE = True
CHANNEL = 12           # which channel to plot
```

## Multiple channels (shortlist)
```python
SINGLE_CHANNEL_MODE = False
CHANNEL_LIST = [0, 4, 12, 16]   # only plot these channels
AMPLITUDE_IN_MILLIVOLTS = True  # y-axis in mV if better for EMG scale.
```

## Side‑by‑side comparison
Plot the same channel twice with different filters/limits by saving two images (or open two windows) rather than subplots to keep things simple.

## Tips
- Large CSVs: consider downsampling for preview to avoid slow rendering.
- If the plot looks “flat” in multi-channel mode, the amplitude may be far too large.
