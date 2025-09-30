
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional
import numpy as np

try:
    from scipy import signal  # type: ignore
except Exception:  # pragma: no cover
    signal = None

try:
    import pywt  # type: ignore
except Exception:  # pragma: no cover
    pywt = None


@dataclass
class SegmentationParams:
    # Preprocessing
    use_wavelet_denoise: bool = True
    wavelet: str = "db4"
    wavelet_level: Optional[int] = None

    # Butterworth bandpass
    bandpass_low_hz: float = 20.0
    bandpass_high_hz: float = 450.0
    bandpass_order: int = 4

    # Notch (NZ mains)
    notch_hz: Optional[float] = 50.0
    notch_q: float = 30.0
    notch_harmonics: int = 2

    # Envelope / STE (moving RMS)
    rms_window_ms: float = 100.0

    # Thresholding
    threshold_mode: str = "percentile"   # "percentile" | "zscore"
    percentile: float = 65.0
    z_thresh: float = 2.5

    # Post-processing
    min_duration_ms: float = 200.0
    merge_gap_ms: float = 100.0

    # Channel aggregation
    aggregate: str = "mean_abs"          # "mean_abs" | "max_abs"


def detect_movement_mask(
    emg: np.ndarray,
    fs: float,
    channels: Optional[Iterable[int]] = None,
    params: Optional[SegmentationParams] = None,
    single_segment: bool = False,
) -> np.ndarray:
    if params is None:
        params = SegmentationParams()

    X = _ensure_2d(emg)  # (C, N)
    C, N = X.shape
    if fs <= 0 or N == 0:
        return np.zeros(N, dtype=np.uint8)

    if channels is not None:
        idx = [c for c in channels if 0 <= int(c) < C]
        if len(idx) == 0:
            idx = list(range(C))
        X = X[idx, :]

    # Wavelet denoise (per channel) if available
    if params.use_wavelet_denoise and pywt is not None:
        X = np.vstack([_wavelet_denoise(ch, params.wavelet, params.wavelet_level) for ch in X])

    # Bandpass + optional notch
    if signal is not None:
        X = _butter_bandpass_apply(X, fs, params.bandpass_low_hz, params.bandpass_high_hz, params.bandpass_order)
        if params.notch_hz:
            X = _apply_notch(X, fs, params.notch_hz, params.notch_q, params.notch_harmonics)

    # Envelope via moving RMS of rectified signal
    env = _compute_envelope(X, fs, params.rms_window_ms, params.aggregate)

    # Threshold
    if params.threshold_mode == "percentile":
        thr = float(np.percentile(env, params.percentile))
        active = (env >= thr)
    else:
        mu = float(np.mean(env)); sd = float(np.std(env) + 1e-9)
        active = ((env - mu) / sd >= params.z_thresh)

    # Post-process
    mask = _postprocess_mask(active.astype(np.uint8), fs, params.min_duration_ms, params.merge_gap_ms)

    # Keep only the largest segment if requested
    if single_segment:
        mask = _keep_largest(mask)

    return mask.astype(np.uint8)


def _ensure_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    # Expect (C,N). If samples-first (N,C), transpose.
    if x.shape[0] > x.shape[1]:
        x = x.T
    return x


def _wavelet_denoise(sig: np.ndarray, wavelet: str, level: Optional[int]) -> np.ndarray:
    if pywt is None:
        return sig
    coeffs = pywt.wavedec(sig, wavelet=wavelet, level=level)
    sigma = _mad(coeffs[-1]) / 0.6745 + 1e-12
    uthresh = sigma * np.sqrt(2 * np.log(sig.size))
    den = [coeffs[0]] + [pywt.threshold(c, value=uthresh, mode="soft") for c in coeffs[1:]]
    out = pywt.waverec(den, wavelet)[: sig.size]
    return out


def _mad(x: np.ndarray) -> float:
    return float(np.median(np.abs(x - np.median(x))))


def _butter_bandpass_apply(X: np.ndarray, fs: float, low: float, high: float, order: int) -> np.ndarray:
    nyq = 0.5 * fs
    low_n = max(1e-6, low / nyq)
    high_n = min(0.999999, high / nyq)
    if signal is None or high_n <= low_n:
        return X
    b, a = signal.butter(order, [low_n, high_n], btype="bandpass")
    pad = min(3 * max(len(a), len(b)), X.shape[1] - 1)
    return signal.filtfilt(b, a, X, axis=1, padlen=pad)


def _apply_notch(X: np.ndarray, fs: float, f0: float, q: float, harmonics: int) -> np.ndarray:
    if signal is None:
        return X
    out = X
    for k in range(1, max(1, harmonics) + 1):
        w0 = (f0 * k) / (fs / 2.0)  # normalized
        if 0 < w0 < 1:
            b, a = signal.iirnotch(w0, q)
            pad = min(3 * max(len(a), len(b)), out.shape[1] - 1)
            out = signal.filtfilt(b, a, out, axis=1, padlen=pad)
    return out


def _compute_envelope(X: np.ndarray, fs: float, window_ms: float, aggregate: str) -> np.ndarray:
    rect2 = X * X
    win = max(1, int(round(window_ms * fs / 1000.0)))
    kernel = np.ones(win, dtype=float) / float(win)
    # moving average of squared signal, then sqrt
    rms = np.sqrt(np.maximum(1e-12, np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), 1, rect2)))
    if aggregate == "max_abs":
        env = np.max(rms, axis=0)
    else:
        env = np.mean(rms, axis=0)
    return env


def _postprocess_mask(mask: np.ndarray, fs: float, min_dur_ms: float, merge_gap_ms: float) -> np.ndarray:
    N = mask.size
    if N == 0:
        return mask
    min_len = max(1, int(round(min_dur_ms * fs / 1000.0)))
    gap = max(1, int(round(merge_gap_ms * fs / 1000.0)))

    # segments
    segs = []
    i = 0
    while i < N:
        if mask[i]:
            j = i
            while j < N and mask[j]:
                j += 1
            segs.append((i, j))
            i = j
        else:
            i += 1

    # merge short gaps
    merged = []
    for seg in segs:
        if not merged:
            merged.append(seg)
        else:
            ps, pe = merged[-1]
            s, e = seg
            if s - pe <= gap:
                merged[-1] = (ps, e)
            else:
                merged.append(seg)

    # drop short segments
    merged = [(s, e) for (s, e) in merged if (e - s) >= min_len]

    out = np.zeros(N, dtype=np.uint8)
    for s, e in merged:
        out[s:e] = 1
    return out


def _keep_largest(mask: np.ndarray) -> np.ndarray:
    N = mask.size
    best = (0, 0)
    i = 0
    while i < N:
        if mask[i]:
            j = i
            while j < N and mask[j]:
                j += 1
            if (j - i) > (best[1] - best[0]):
                best = (i, j)
            i = j
        else:
            i += 1
    out = np.zeros_like(mask, dtype=np.uint8)
    s, e = best
    if e > s:
        out[s:e] = 1
    return out

