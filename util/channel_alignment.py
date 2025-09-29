"""Channel alignment utility for EMG-only frames.

Provides a heuristic to locate the sample-counter channel in recent frames
and compute the byte offset needed to align frame boundaries in a raw byte
buffer from the Muovi/SyncStation stream.
"""

import numpy as np


def simple_alignment(data_buffer):
    """Estimate frame alignment offset from the tail of a byte buffer.

    Takes the last 10 frames (assumed 88 bytes per frame for EMG-only) and
    decodes EMG + auxiliary channels to locate a monotonically increasing
    sample-counter channel. Returns the number of bytes to trim from the end
    so that the buffer ends at a frame boundary.

    Args:
        data_buffer (bytes | bytearray | memoryview): Raw incoming byte buffer.

    Returns:
        int: Byte offset to drop from the end of `data_buffer` (0 if not found).

    Notes:
        - Assumes 16-bit EMG samples (2 bytes) and 6 auxiliary channels at the
          end of each 88-byte frame.
        - Detects a counter by checking consecutive +1 increments across the
          first three samples; uses a second channel (+6) to disambiguate.
    """

    num_bytes = 88
    samples = data_buffer[-(num_bytes * 10):]
    data = np.zeros((45, 10))

    temp_array = np.frombuffer(samples, dtype=np.uint8)
    temp = np.reshape(temp_array, (-1, 88)).T  # dynamic reshape
    # Processing data
    chan_ready = 1
    for DevId in range(16):
        if DevId == 0:
            ch_ind = np.arange(0, 38 * 2, 2)
            data_sub_matrix = temp[ch_ind].astype(np.int32) * 256 + temp[ch_ind + 1].astype(np.int32)

            # Search for the negative values and make the two's complement
            ind = np.where(data_sub_matrix >= 32768)
            data_sub_matrix[ind] = data_sub_matrix[ind] - 65536

            data[chan_ready:chan_ready + 38, :] = data_sub_matrix


            del ch_ind
            del ind
            del data_sub_matrix
            chan_ready += 38
    aux_starting_byte = 88 - (6 * 2)
    ch_ind = np.arange(aux_starting_byte, aux_starting_byte + 12, 2)
    data_sub_matrix = temp[ch_ind].astype(np.int32) * 256 + temp[ch_ind + 1].astype(np.int32)

    # Search for the negative values and make the two's complement
    ind = np.where(data_sub_matrix >= 32768)
    data_sub_matrix[ind] = data_sub_matrix[ind] - 65536

    data[chan_ready:chan_ready + 6, :] = data_sub_matrix
    for i, D in enumerate(data):
        if D[0] == D[1] - 1 and D[1] == D[2] - 1:
            plus6 = data[(i+6)%44]
            if plus6[0] == plus6[1] - 1 and plus6[1] == plus6[2] - 1:
                return (38-i)%44 * 2
            else:
                return (44-i)%44 * 2
    print("Failed to find sample counter channel")
    return 0




