"""Validation utilities for checking EMG/EEG data streams.

Provides simple functions to confirm that active channels from EMG and EEG
devices contain non-zero values, ensuring that the hardware is connected
and streaming before starting a session.
"""

from config import Config
def validate_data(data, use_emg, use_eeg):
    """Check whether EMG/EEG data arrays contain valid (non-zero) samples.

    Args:
        data (np.ndarray): 2D array of channel data, shaped (num_channels, num_samples).
        use_emg (bool): Whether EMG device(s) are enabled.
        use_eeg (bool): Whether EEG device(s) are enabled.

    Returns:
        bool: True if all enabled devices provide non-zero data on their first channel,
        False otherwise.

    Notes:
        - Uses the channel indices defined in `Config` to locate EMG and EEG streams.
        - Prints a confirmation message when valid data is found.
    """

    config = Config(use_emg, use_eeg)
    if use_emg and not zero_check(data[config.MUOVI_EMG_CHANNELS[0]]):
        return False
    if use_eeg and not zero_check(data[config.MUOVI_PLUS_EEG_CHANNELS[0]]):
        return False
    print("Data found for all selected devices")
    return True

def zero_check(channel):
    """Return True if a channel contains at least one non-zero sample.

    Args:
        channel (Iterable[int | float]): Sequence of samples for a single channel.

    Returns:
        bool: True if any sample is non-zero, False if all samples are zero.
    """

    return not all(sample == 0 for sample in channel)