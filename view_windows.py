"""Plot a given channel from windowed data stored in HDF5.

You MUST specify the axis order of your dataset so we can permute it to:
    (n_windows, window_size, n_channels)

Examples of AXIS_ORDER (index positions of those axes IN YOUR FILE):
- If your dataset is already (n_windows, window_size, n_channels): AXIS_ORDER = (0, 1, 2)
- If it is (n_windows, n_channels, window_size):                 AXIS_ORDER = (0, 2, 1)
- If it is (window_size, n_channels, n_windows):                 AXIS_ORDER = (2, 0, 1)
- If it is (n_channels, window_size, n_windows):                 AXIS_ORDER = (2, 1, 0)

After permuting, we select [start:end, :, channel] and plot.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")

# -------- User settings --------
FILE_WINDOWS   = "data/trial_10_processed_emg.h5"
DATASET_NAME   = "windowed_data"

# Tell the script how your dataset is laid out IN THE FILE:
# AXIS_ORDER = (idx_of_n_windows, idx_of_window_size, idx_of_n_channels)
AXIS_ORDER     = (0, 1, 2)      # CHANGE THIS to match your file layout

CHANNEL        = 19             # 0-based channel index to plot
START_WINDOW   = 0              # first window index
NUM_WINDOWS    = 11             # how many windows to plot
OVERLAY        = True           # True = one axis, False = stacked subplots
NORMALIZE_PER_WINDOW = False    # visual-only: rescales each window to [-1,1]
# --------------------------------


def _normalize_row(x: np.ndarray) -> np.ndarray:
    rng = np.max(x) - np.min(x)
    if rng < 1e-12:
        return x * 0.0
    return (x - np.mean(x)) / rng


def plot_windows_h5(
    file_path: str,
    dataset_name: str,
    axis_order: tuple[int, int, int],
    channel: int,
    start: int,
    num: int,
    overlay: bool,
    normalize_per_window: bool,
):
    with h5py.File(file_path, "r") as f:
        if dataset_name not in f:
            raise KeyError(f"Dataset '{dataset_name}' not found. Available: {list(f.keys())}")
        arr = f[dataset_name][:]
    print(f"[info] Loaded '{dataset_name}' with raw shape {arr.shape}, dtype={arr.dtype}")

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D dataset, got {arr.ndim}D with shape {arr.shape}")

    # Validate axis_order
    ao = tuple(axis_order)
    if sorted(ao) != [0, 1, 2]:
        raise ValueError(f"AXIS_ORDER must be a permutation of (0,1,2); got {axis_order}")

    # Permute to (n_windows, window_size, n_channels)
    permuted = np.transpose(arr, ao)
    n_windows, window_size, n_channels = permuted.shape
    print(f"[info] Interpreting as (n_windows={n_windows}, window_size={window_size}, n_channels={n_channels})")

    if not (0 <= channel < n_channels):
        raise ValueError(f"CHANNEL {channel} out of range [0, {n_channels-1}]")

    end = min(start + num, n_windows)
    if start >= end:
        raise ValueError(f"Empty selection: start={start}, num={num}, n_windows={n_windows}")

    selected = permuted[start:end, :, channel]  # shape: (num, window_size)
    print(f"[info] Selected windows [{start}:{end}) → selected.shape={selected.shape}")

    if normalize_per_window:
        selected = np.stack([_normalize_row(w) for w in selected], axis=0)

    if overlay:
        plt.figure(figsize=(15, 5))
        for i, win in enumerate(selected, start=start):
            plt.plot(win, alpha=0.85, linewidth=1.0, label=f"W{i}")
        plt.title(f"Channel {channel} — Windows {start}..{end-1} (overlay)")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude" + (" (norm)" if normalize_per_window else ""))
        if selected.shape[0] <= 15:
            plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        rows = selected.shape[0]
        fig, axes = plt.subplots(nrows=rows, ncols=1, figsize=(15, 2*rows), sharex=True)
        if rows == 1:
            axes = [axes]
        for idx, (ax, win) in enumerate(zip(axes, selected), start=start):
            ax.plot(win, linewidth=1.0)
            ax.set_ylabel(f"W{idx}")
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel("Sample")
        fig.suptitle(f"Channel {channel} — Windows {start}..{end-1}")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    plot_windows_h5(
        FILE_WINDOWS,
        DATASET_NAME,
        AXIS_ORDER,
        CHANNEL,
        START_WINDOW,
        NUM_WINDOWS,
        OVERLAY,
        NORMALIZE_PER_WINDOW,
    )
