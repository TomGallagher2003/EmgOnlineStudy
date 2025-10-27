from sklearn.preprocessing import MinMaxScaler
import numpy as np


def normalise_data(data: np.ndarray) -> np.ndarray:
    """Min–max normalise a 3D windowed array feature-wise to [0, 1].

    This applies scikit-learn’s :class:`~sklearn.preprocessing.MinMaxScaler` over
    the **channel/features dimension** after flattening the time and window
    axes. Concretely, the input of shape ``(n_windows, window_size, n_channels)``
    is reshaped to ``(n_windows * window_size, n_channels)``, scaled per column
    to the range ``[0, 1]``, and then reshaped back to the original 3D shape.

    Args:
        data: Array of shape ``(n_windows, window_size, n_channels)`` containing
            windowed samples. Each column (channel) is scaled independently.

    Returns:
        np.ndarray: Normalised array with the **same shape** as ``data``.

    Notes:
        - The scaling parameters are computed **from this batch only** (no fit
          reuse). For consistent train/test transforms, persist and reuse the
          fitted scaler externally.
        - Values constant across all samples in a channel will be mapped to 0.0.
    """
    scaler = MinMaxScaler()
    # Flatten to (total_samples, n_channels)
    reshaped = data.reshape(-1, data.shape[-1])
    normed = scaler.fit_transform(reshaped)
    return normed.reshape(data.shape)
