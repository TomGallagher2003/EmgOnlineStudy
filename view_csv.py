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


FILENAME = "data/trial_12_raw_emg.csv"                           # Set your file name here
SINGLE_CHANNEL_MODE = True
CHANNEL = 12

START_CHANNEL = 10
NUM_CHANNELS = 5

CHANNEL_LIST = []




AMPLITUDE_IN_MILLIVOLTS = 0.5               # Only affects multi-channel mode. Adjust as necessary


MICRO_VOLTS = False
if FILENAME.split("\\")[-1].startswith("eeg"):
    MICRO_VOLTS = True

def plot_file(file_path, channel_list=[]):
    """Plot multiple channels from a CSV signal file in stacked subplots.

Loads the CSV at `file_path`, transposes to (channels, samples), optionally
selects a subset of channels, applies EEG microvolt scaling if the filename
starts with 'eeg', and renders each channel on its own axis with shared X.

Args:
    file_path (str | Path): Path to the CSV file (channels in columns or rows;
        function transposes to channel-major).
    channel_list (Iterable[int], optional): Zero-based indices of channels to
        include. If empty, all channels are plotted.

Notes:
    - When `MICRO_VOLTS` is True (filename starts with 'eeg'), data is multiplied
      by 1e3 to convert mV→µV for display.
    - The Y-range of each subplot is clamped to ±`AMPLITUDE_IN_MILLIVOLTS`
      (interpreted as mV or µV depending on mode).
"""


    data = np.loadtxt(file_path, delimiter=',')
    data = data.transpose()
    if len(channel_list) > 0:
        data = data[channel_list]


    amplitude = AMPLITUDE_IN_MILLIVOLTS
    if MICRO_VOLTS:
        data = data * 1e3
        amplitude = amplitude * 1e3
    print(data.shape)


    plt.clf()
    fig, axes = plt.subplots(nrows=data.shape[0], ncols=1, figsize=(16, 16), sharex=True)
    fig.suptitle(f'file: {file_path}', fontsize=16)
    X = 0

    for j, emg_signal in enumerate(data):
        axes[j].set_ylim(-1 * amplitude, amplitude)
        axes[j].set_yticks([])
        axes[j].set_xticks([])
        axes[j].plot(emg_signal, label=f'Channel {j + 1}')



    plt.show()
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
    unit_label = "mV"
    if max([max(x) for x in data[5:20]]) > 500:
        unit_label = "raw input"
    elif MICRO_VOLTS:
        data = data * 1e3
        unit_label = "µV"
    
    

    plt.clf()
    plt.figure(figsize=(15, 5))
    plt.ylabel(unit_label)

    plt.plot(data[channel-1])



    plt.show()


# Entry point: selects plotting mode based on flags/args and renders the figure.
if __name__ == '__main__':

    if SINGLE_CHANNEL_MODE:
        plot_channel(FILENAME, CHANNEL)
    elif len(CHANNEL_LIST) > 0:
        plot_file(FILENAME,  CHANNEL_LIST)
    elif START_CHANNEL and NUM_CHANNELS:
        plot_file(FILENAME, range(START_CHANNEL, START_CHANNEL + NUM_CHANNELS))
    else:
        plot_file(FILENAME)
