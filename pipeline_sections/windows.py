import numpy as np

def window_data(data: np.ndarray, sample_size: int = 512, step_size: int = 512) -> np.ndarray:
    """
    Slice data into fixed-length windows using a sliding window.

    Args:
        data: (n_samples, n_channels) array
        sample_size: number of samples per window
        step_size: hop length between windows

    Returns:
        windows: (n_windows, sample_size, n_channels) array
    """
    segments = []
    n_samples, _ = data.shape

    start_index = 0
    while start_index + sample_size <= n_samples:
        end_index = start_index + sample_size
        segments.append(data[start_index:end_index])
        start_index += step_size

    return np.stack(segments) if segments else np.empty((0, sample_size, data.shape[1]))
