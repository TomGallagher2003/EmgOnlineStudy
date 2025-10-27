import numpy as np

def window_data(data: np.ndarray, sample_size: int = 512, overlap: int = 0) -> np.ndarray:
    """Slice a multichannel signal into fixed-length, possibly overlapping windows.

    Uses a sliding window with hop length ``step_size = sample_size - overlap``.
    Windows are emitted only when a **full** window fits inside ``data``.

    Args:
        data:
            Array of shape ``(n_samples, n_channels)``.
        sample_size:
            Number of samples per window (window length).
        overlap:
            Number of samples that consecutive windows share
            (must satisfy ``0 <= overlap < sample_size``).

    Returns:
        np.ndarray:
            Window tensor of shape ``(n_windows, sample_size, n_channels)``.
            If fewer than ``sample_size`` samples are available, returns an empty
            array with shape ``(0, sample_size, n_channels)``.

    Notes:
        - This function does **not** pad; trailing samples that cannot form a full
          window are discarded.
        - No copying is guaranteed—NumPy may copy or view depending on strides.
    """
    step_size = sample_size - overlap
    segments = []
    n_samples, _ = data.shape

    start_index = 0
    while start_index + sample_size <= n_samples:
        end_index = start_index + sample_size
        segments.append(data[start_index:end_index])
        start_index += step_size

    return np.stack(segments) if segments else np.empty((0, sample_size, data.shape[1]))


def window_labels(labels: np.ndarray, sample_size: int = 512, overlap: int = 0) -> np.ndarray:
    """Window a 1-D label sequence to align with windowed data.

    Uses the same sliding-window scheme as :func:`window_data` with hop length
    ``step_size = sample_size - overlap``. Only full windows are returned.

    Args:
        labels:
            1-D array of length ``n_samples`` containing integer/float labels.
        sample_size:
            Number of samples per window.
        overlap:
            Number of samples shared by consecutive windows
            (must satisfy ``0 <= overlap < sample_size``).

    Returns:
        np.ndarray:
            Array of shape ``(n_windows, sample_size)`` containing label windows.
            Returns an empty array with shape ``(0, sample_size)`` if not enough
            samples exist for one full window.

    Notes:
        - Keep ``sample_size`` and ``overlap`` identical to the values passed to
          :func:`window_data` to maintain alignment.
    """
    step_size = sample_size - overlap
    labels = np.asarray(labels)
    segments = []
    n_samples = labels.shape[0]

    start = 0
    while start + sample_size <= n_samples:
        end = start + sample_size
        segments.append(labels[start:end])
        start += step_size

    return np.stack(segments) if segments else np.empty((0, sample_size), dtype=labels.dtype)
