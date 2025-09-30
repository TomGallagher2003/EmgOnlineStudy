"""Signal filtering utilities for EMG/EEG data.

This module provides reusable Butterworth filters for high-pass,
low-pass, band-pass, and band-stop filtering, along with a preprocessing
pipeline tailored for EEG signals that applies a high-pass, notch filters
to remove line noise, and a low-pass cutoff.

Default sampling frequency is 500 Hz.
"""

from scipy.signal import butter, filtfilt
import numpy as np

FS = 500.0  # sampling frequency (Hz)


def highpass_filter(data: np.ndarray, cutoff: float = 0.1, order: int = 4, fs: float = FS) -> np.ndarray:
    """Apply a Butterworth high-pass filter.

    Args:
        data (np.ndarray): 2D array of signals (channels x samples).
        cutoff (float, optional): Cutoff frequency in Hz. Defaults to 0.1.
        order (int, optional): Filter order. Defaults to 4.
        fs (float, optional): Sampling frequency in Hz. Defaults to FS.

    Returns:
        np.ndarray: Filtered signals.
    """
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="high")
    return filtfilt(b, a, data, axis=1)


def lowpass_filter(data: np.ndarray, cutoff: float, order: int = 4, fs: float = FS) -> np.ndarray:
    """Apply a Butterworth low-pass filter.

    Args:
        data (np.ndarray): 2D array of signals (channels x samples).
        cutoff (float): Cutoff frequency in Hz.
        order (int, optional): Filter order. Defaults to 4.
        fs (float, optional): Sampling frequency in Hz. Defaults to FS.

    Returns:
        np.ndarray: Filtered signals.
    """
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, data, axis=1)


def bandpass_filter(data: np.ndarray, low: float, high: float, order: int = 4, fs: float = FS) -> np.ndarray:
    """Apply a Butterworth band-pass filter.

    Args:
        data (np.ndarray): 2D array of signals (channels x samples).
        low (float): Low cutoff frequency in Hz.
        high (float): High cutoff frequency in Hz.
        order (int, optional): Filter order. Defaults to 4.
        fs (float, optional): Sampling frequency in Hz. Defaults to FS.

    Returns:
        np.ndarray: Filtered signals.
    """
    nyq = 0.5 * fs
    low_norm = low / nyq
    high_norm = high / nyq
    b, a = butter(order, [low_norm, high_norm], btype="band")
    return filtfilt(b, a, data, axis=1)


def bandstop_filter(data: np.ndarray, low: float, high: float, order: int = 2, fs: float = FS) -> np.ndarray:
    """Apply a Butterworth band-stop (notch) filter.

    Args:
        data (np.ndarray): 2D array of signals (channels x samples).
        low (float): Low cutoff frequency in Hz.
        high (float): High cutoff frequency in Hz.
        order (int, optional): Filter order. Defaults to 2.
        fs (float, optional): Sampling frequency in Hz. Defaults to FS.

    Returns:
        np.ndarray: Filtered signals.
    """
    nyq = 0.5 * fs
    low_norm = low / nyq
    high_norm = high / nyq
    b, a = butter(order, [low_norm, high_norm], btype="bandstop")
    return filtfilt(b, a, data, axis=1)


def remove_line_noise(data: np.ndarray, fs: float = FS) -> np.ndarray:
    """Remove line noise at 50 Hz (and optionally 100 Hz) using band-stop filters.

    Args:
        data (np.ndarray): 2D array of signals (channels x samples).
        fs (float, optional): Sampling frequency in Hz. Defaults to FS.

    Returns:
        np.ndarray: Filtered signals with line noise attenuated.
    """
    data = bandstop_filter(data, 49.9, 50.1, order=2, fs=fs)   # 50 Hz
    # data = bandstop_filter(data, 99, 101, order=2, fs=fs)   # 100 Hz (disabled by default)
    return data

def notch_filter(data: np.ndarray, val: float, fs: float = FS) -> np.ndarray:
    data = bandstop_filter(data, val - 0.1, val + 0.1, order=2, fs=fs)
    return data


def preprocess_eeg(data: np.ndarray, fs: float = FS) -> np.ndarray:
    """Apply preprocessing pipeline to EEG data.

    Steps:
        1. High-pass filter at 0.3 Hz (2nd order).
        2. Band-stop at 50 Hz (line noise).
        3. Low-pass filter at 70 Hz (2nd order).

    Args:
        data (np.ndarray): 2D array of EEG signals (channels x samples).
        fs (float, optional): Sampling frequency in Hz. Defaults to FS.

    Returns:
        np.ndarray: Preprocessed EEG signals.
    """
    data = highpass_filter(data, cutoff=0.3, order=2, fs=fs)
    data = remove_line_noise(data, fs=fs)
    data = lowpass_filter(data, cutoff=70, order=2, fs=fs)
    return data


def selective_filter(filters, data):
    """
    filters: list of dicts like
       {"type": "None"|"Pass"|"Notch", "lower": float|None, "upper": float|None, "center": float|None}
    Apply in order: A then B then C.
    - Pass:
        * lower only -> high-pass
        * upper only -> low-pass
        * both -> band-pass (lower < upper guaranteed by UI)
    - Notch:
        * center required
    - None: skip
    """
    out = data
    for f in filters:
        t = f["type"]
        if t == "Pass":
            lo, up = f["lower"], f["upper"]
            if lo is not None and up is not None:
                out = bandpass_filter(out, lo, up)
            elif lo is not None:
                out = highpass_filter(out, lo)
            elif up is not None:
                out = lowpass_filter(out, up)
        elif t == "Notch":
            if f["center"] is not None:
                out = notch_filter(out, f["center"])
    return out
