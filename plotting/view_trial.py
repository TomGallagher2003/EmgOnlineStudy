"""Simple plotting utilities for EMG/EEG CSV files.

Provides helpers to visualize one or more channels from a comma-separated
signal file. If the filename begins with 'eeg', values are treated as microvolts
(µV) and scaled accordingly. Multi-channel plots normalize Y-limits using the
`AMPLITUDE_IN_MILLIVOLTS` setting (mV by default).
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from config import Config

matplotlib.use('TkAgg')
TRIAL = 98

FILENAME = f"data/trial_{TRIAL}_raw_emg.csv"
LABEL_FILENAME = f"data/trial_{TRIAL}_auto_label.csv"
CHANNEL = 12

MICRO_VOLTS = True



def plot_channel(file_path, channel=1):
    """Plot a single channel from a CSV signal file.

Loads and transposes the CSV at `file_path`, applies unit heuristics, and plots
the specified 1-based `channel`. If the maximum value across channels 6–20
(1-based) is > 500, the Y-label is set to 'raw input'; otherwise:
- If EEG filename (starts with 'eeg'), data is scaled to µV and label 'µV'
- Else label defaults to 'mV'

Args:
    file_path (str | Path): Path to the CSV file.
    channel (int, default=1): 1-based channel index to visualize.

Notes:
    - Uses simple heuristics to choose the unit label; adjust for your pipeline
      if raw counts vs. calibrated units differ.
"""

    data = np.loadtxt(file_path, delimiter=',')
    data = data.transpose()
    label = np.loadtxt("../" + LABEL_FILENAME, delimiter=',')

    unit_label = "mV"
    if max([max(x) for x in data[5:20]]) > 500:
        unit_label = "raw input"
    elif MICRO_VOLTS:
        data = data * 1e3
        unit_label = "µV"

    plt.clf()
    plt.figure(figsize=(15, 5))
    plt.ylabel(unit_label)
    plt.fill_between(np.arange(len(label)), min(data[channel-1]), max(data[channel-1]), where=[l!=0 for l in label], interpolate=True, alpha=0.3)

    plt.plot(data[channel - 1])
    plt.show()
import numpy as np

def mask_to_segments(mask: np.ndarray):
    """
    Convert a 1D binary mask into start/end index pairs
    for contiguous non-zero regions.

    Args:
        mask: 1D array-like of 0/1 values

    Returns:
        List of (start, end) index tuples (end is inclusive)
    """
    mask = np.asarray(mask, dtype=bool)
    # Find rising and falling edges
    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends   = np.where(diff == -1)[0]

    # Handle edge cases if mask starts/ends inside a region
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, len(mask) - 1]

    return list(zip(starts, ends))


# Entry point: selects plotting mode based on flags/args and renders the figure.
if __name__ == '__main__':

        plot_channel("../" + FILENAME, CHANNEL)
