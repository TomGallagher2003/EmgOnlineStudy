"""Processing functions for decoding raw EMG/EEG data streams.

This module converts raw byte buffers from the Muovi devices into
scaled EMG/EEG channel values in millivolts. Handles both EMG and
EEG devices, applies two’s complement conversion, scaling by gain
ratios, and filtering for EEG signals.
"""

import numpy as np
from config import Config
from util.filter_helpers import preprocess_eeg


def process(config, temp, data, tot_num_byte, chan_ready):
    """Decode and process raw EMG/EEG bytes into channel data.

    Args:
        config (Config): Configuration object containing channel maps,
            gain ratios, device enables, and mode flags.
        temp (np.ndarray): 1D array of raw byte values for one frame.
        data (np.ndarray): Output 2D array where processed channel
            values are written.
        tot_num_byte (int): Total number of bytes expected in the frame.
        chan_ready (int): Starting index in `data` for the next block
            of processed channels.

    Returns:
        np.ndarray: Updated `data` array with processed EMG/EEG and
            auxiliary channel values.

    Notes:
        - EMG data are 16-bit samples (2 bytes/channel).
        - EEG data are 24-bit samples (3 bytes/channel).
        - EEG channels are filtered with a 0.3–70 Hz bandpass and
          50 Hz notch filter via `preprocess_eeg`.
        - Both EMG and EEG signals are converted to millivolts.
    """

    # Processing data
    for DevId in range(16):
        if config.DEVICE_EN[DevId] == 1:
            if config.EMG[DevId] == 1:
                # EMG CASE
                ch_ind = np.arange(0, 32 * 2, 2)
                ch_ind_aux = np.arange(32 * 2, 38 * 2, 2)
                data_sub_matrix = temp[ch_ind].astype(np.int32) * 256 + temp[ch_ind + 1].astype(np.int32)
                data_sub_matrix_aux = temp[ch_ind_aux].astype(np.int32) * 256 + temp[ch_ind_aux + 1].astype(np.int32)

                # Search for the negative values and make the two's complement (not on aux)
                ind = np.where(data_sub_matrix >= 32768)
                data_sub_matrix[ind] = data_sub_matrix[ind] - 65536

                # converting raw volts to mV using the ratios from the documentation
                data_sub_matrix = data_sub_matrix * config.GAIN_RATIOS[config.EMG_MODE] * 1e3

                data[chan_ready:chan_ready + 32, :] = data_sub_matrix
                data[chan_ready + 32:chan_ready + 38, :] = data_sub_matrix_aux

            else:
                # EEG CASE
                start = config.MUOVI_PLUS_EEG_CHANNELS[0] * 2
                ch_ind = np.arange(start, start + 64 * 3, 3)
                ch_ind_aux = np.arange(start + 64 * 3, start + 64 * 3 + 6 * 3, 3)
                data_sub_matrix = temp[ch_ind].astype(np.int32) * 65536 + temp[ch_ind + 1].astype(np.int32) * 256 + \
                                  temp[ch_ind + 2].astype(np.int32)

                data_sub_matrix_aux = temp[ch_ind_aux].astype(np.int32) * 65536 + temp[ch_ind_aux + 1].astype(np.int32) * 256 + \
                                  temp[ch_ind_aux + 2].astype(np.int32)

                # Search for the negative values and make the two's complement
                ind = np.where(data_sub_matrix >= 8388608)
                data_sub_matrix[ind] = data_sub_matrix[ind] - 16777216


                # converting raw volts to mV using the ratios from the documentation
                data_sub_matrix = data_sub_matrix * config.GAIN_RATIOS[config.EEG_MODE] * 1e3


                data[chan_ready:chan_ready + 64, :] = data_sub_matrix
                data[chan_ready + 64:chan_ready + 70, :] = data_sub_matrix_aux

            del ch_ind
            del ind
            del data_sub_matrix
            chan_ready += config.NUM_CHAN[DevId]

    aux_starting_byte = tot_num_byte - (6 * 2)
    ch_ind = np.arange(aux_starting_byte, aux_starting_byte + 12, 2)
    data_sub_matrix = temp[ch_ind].astype(np.int32) * 256 + temp[ch_ind + 1].astype(np.int32)
    # Search for the negative values and make the two's complement
    # ind = np.where(data_sub_matrix >= 32768)
    # data_sub_matrix[ind] = data_sub_matrix[ind] - 65536
    data[chan_ready:chan_ready + 6, :] = data_sub_matrix
    del data_sub_matrix

    return data