import numpy as np


def mask_to_segments(mask: np.ndarray):
    """
    Convert a 1D binary mask into start/end index pairs
    for contiguous non-zero regions.

    Args:
        mask: 1D array-like of 0/1 values

    Returns:
        List of (start, end) index tuples (end is inclusive)
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    # Find rising and falling edges
    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends   = np.where(diff == -1)[0]

    # Handle edge cases if mask starts/ends inside a region
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, len(mask) - 1]

    return list(zip(starts, ends))