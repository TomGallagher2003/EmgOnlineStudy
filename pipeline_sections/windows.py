import numpy as np

def window_data(data: np.ndarray, sample_size: int = 512, overlap: int = 0) -> np.ndarray:
    """
    Slice data into fixed-length windows using a sliding window.

    Args:
        data: (n_samples, n_channels) array
        sample_size: number of samples per window
        step_size: hop length between windows

    Returns:
        windows: (n_windows, sample_size, n_channels) array
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
    """
    Slice a 1D label array into fixed-length windows.

    Args:
        labels: (n_samples,) array of labels
        sample_size: number of samples per window
        step_size: hop length between windows

    Returns:
        windows: (n_windows, sample_size) array of label windows
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


