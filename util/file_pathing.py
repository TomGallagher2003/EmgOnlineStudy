"""File I/O utilities for saving EMG/EEG session data.

This module provides functions to:
- Create subject-specific directory structures for storing recordings.
- Save channel data and labels into both CSV and HDF5 formats,
  organized by subject, data type, and exercise set.

Directories are created automatically if they do not already exist.
"""

from pathlib import Path
import os
import numpy as np
import h5py
from datetime import datetime


def make_subject_directory(base_path, subject_id, exercise_set,
                           use_emg: bool = True,
                           use_eeg: bool = True,
                           save_counters: bool = True) -> Path:
    """Create subject folder structure for recordings.

    Args:
        base_path (str or Path): Root directory where data is stored.
        subject_id (str or int): Subject identifier (used as folder name).
        exercise_set (str): Exercise set indicator ("A", "B", or "AB").
        use_emg (bool, optional): If True, create EMG directories. Defaults to True.
        use_eeg (bool, optional): If True, create EEG directories. Defaults to True.
        save_counters (bool, optional): If True, create counters directories. Defaults to True.

    Returns:
        Path: Path to the created subject directory.

    Directory Layout:
        <base_path>/<subject_id>/
            emg/EA/{csv,hdf5}/
            emg/EB/{csv,hdf5}/
            eeg/EA/{csv,hdf5}/
            eeg/EB/{csv,hdf5}/
            counters/EA/{csv,hdf5}/
            counters/EB/{csv,hdf5}/
    """
    dir_path = Path(base_path) / str(subject_id)
    dir_path.mkdir(parents=True, exist_ok=True)

    want_A = exercise_set in ("A", "AB")
    want_B = exercise_set in ("B", "AB")

    def make_branch(kind: str):
        if want_A:
            (dir_path / kind / "EA" / "csv").mkdir(parents=True, exist_ok=True)
            (dir_path / kind / "EA" / "hdf5").mkdir(parents=True, exist_ok=True)
        if want_B:
            (dir_path / kind / "EB" / "csv").mkdir(parents=True, exist_ok=True)
            (dir_path / kind / "EB" / "hdf5").mkdir(parents=True, exist_ok=True)

    if use_emg:
        make_branch("emg")
    if use_eeg:
        make_branch("eeg")
    if save_counters:
        make_branch("counters")

    return dir_path


def save_channels(base_path, subject_id, type_string, group, perform_time,
                  suffix, data, labels, save_h5: bool = True, date_str: str = None) -> None:
    """Save EMG/EEG/counter channel data and labels to disk.

    Args:
        base_path (str or Path): Root directory where subject data is stored.
        subject_id (str or int): Subject identifier.
        type_string (str): Data type ("emg", "eeg", or "counters").
        group (str): Exercise group ("EA" or "EB").
        perform_time (float): Duration of the movement in seconds.
        suffix (str): Suffix string for distinguishing recordings.
        data (np.ndarray): 2D array of recorded data (channels x samples).
        labels (np.ndarray): 2D array of corresponding labels.
        save_h5 (bool, optional): If True, also save HDF5 file. Defaults to True.
        date_str (str, optional): Date string for filenames. If None, uses today’s date (dd-mm).

    Saves:
        - CSV files for data and labels.
        - Optional HDF5 file with datasets "<type>_data" and "<type>_label".

    Filenames include the date, perform time in ms, and provided suffix.
    """
    date_str = date_str or datetime.today().strftime('%d-%m')
    perform_ms = int(perform_time * 1000)
    stem = f"{type_string}_data_{date_str}_{perform_ms}ms_{suffix}"

    root = Path(base_path) / str(subject_id) / type_string / group
    root.mkdir(parents=True, exist_ok=True)

    csv_data = root / "csv" / f"{stem}.csv"
    csv_label = root / "csv" / f"{type_string}_label_{date_str}_{perform_ms}ms_{suffix}.csv"
    h5_path = root / "hdf5" / f"{stem}.h5"

    np.savetxt(csv_data, data.T, delimiter=",")
    np.savetxt(csv_label, labels.T, delimiter=",")

    if save_h5:
        h5_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(h5_path, "w") as hf:
            hf.create_dataset(f"{type_string}_data", data=data.T)
            hf.create_dataset(f"{type_string}_label", data=labels)
