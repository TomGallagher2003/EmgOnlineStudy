from sklearn.preprocessing import MinMaxScaler
import numpy as np


def normalise_data(data: np.ndarray) -> np.ndarray:
    """
    Normalize data for a single movement (all samples same class).

    Args:
        data: (n_windows, window_size, n_channels)

    Returns:
        Normalized data with same shape
    """
    scaler = MinMaxScaler()
    # Flatten to (total_samples, n_channels)
    reshaped = data.reshape(-1, data.shape[-1])
    normed = scaler.fit_transform(reshaped)
    return normed.reshape(data.shape)
