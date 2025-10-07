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

def window_labels(labels: np.ndarray, sample_size: int = 512, step_size: int = 512, reduce: str = "mode"):
    """
    Slice a 1D label array into windows and collapse each window to a single label.

    Args:
        labels:       (n_samples,) array of ints/floats
        sample_size:  number of samples per window
        step_size:    hop length between windows
        reduce:       how to collapse a window to one label:
                      - "mode": most frequent value (default)
                      - "center": label at the center of the window
                      - "first": first element
                      - "last": last element

    Returns:
        win_labels: (n_windows,) array of reduced labels
    """
    labels = np.asarray(labels)
    n_samples = labels.shape[0]
    out = []

    i = 0
    while i + sample_size <= n_samples:
        wlab = labels[i:i + sample_size]

        if reduce == "mode":
            vals, counts = np.unique(wlab, return_counts=True)
            lbl = vals[np.argmax(counts)]
        elif reduce == "center":
            lbl = wlab[(sample_size - 1) // 2]
        elif reduce == "first":
            lbl = wlab[0]
        elif reduce == "last":
            lbl = wlab[-1]
        else:
            raise ValueError(f"Unknown reduce='{reduce}'")

        out.append(lbl)
        i += step_size

    return np.asarray(out)

