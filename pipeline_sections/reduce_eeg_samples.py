def reduce_eeg(eeg_data_2000Hz):
    """Downsample a 2,000 Hz EEG array to ~500 Hz by taking every 4th sample.

    This is a simple decimation (no anti-alias filtering). Use when upstream
    capture is at 2000 Hz and you need a quick reduction for lightweight
    processing or preview.

    Args:
        eeg_data_2000Hz: 1-D or 2-D array-like of samples at 2000 Hz.
            - If 1-D: shape ``(samples,)``.
            - If 2-D: shape ``(samples, channels)`` or ``(channels, samples)`` —
              the operation is applied along the first axis as written.

    Returns:
        A view/slice of the input with stride 4, i.e., ``eeg_data_2000Hz[::4]``.

    Notes:
        - This function does not apply an anti-alias low-pass filter. For
          production downsampling, consider filtering to <250 Hz first.
    """
    return eeg_data_2000Hz[::4]
