"""EMG/EEG recording session controller.

This module provides the `Session` class to configure a SyncStation/EMG device,
send start/stop commands, stream raw frames over TCP, align/parse frames into
per-channel samples, and save EMG/EEG/counter channels with labels for
movement/rest segments. Designed for synchronous EMG/EEG experiments with
optional counter channels and HDF5/CSV persistence.
"""

import gc
import struct
from datetime import datetime
import os

import numpy as np
import time
from util.channel_alignment import simple_alignment
from util.OTB_refactored.configuration_processing import calculate_crc8, validate_config, process_config
from util.file_pathing import save_channels, make_subject_directory
from util.processing import process
from util.socket_handling import SocketHandler
from config import Config

SAMPLE_TOLERANCE = 200

class Session:
    """Manage a recording session for EMG/EEG data acquisition.

        This class encapsulates configuration validation, SyncStation communication,
        streaming/collection of raw bytes, alignment (with or without EEG counters),
        decoding into channel-wise arrays, and saving labeled segments.

        Attributes:
            config (Config): Runtime configuration (devices, channels, sample rate, paths).
            socket_handler (SocketHandler): TCP socket wrapper for the SyncStation/device.
            conf_string (list[int] | None): Mutable configuration command buffer (bytes-as-ints).
            tot_num_byte (int | None): Bytes per frame (all enabled channels in one frame).
            tot_num_chan (int | None): Number of enabled channels in the stream.
            recording (bool): True while actively receiving bytes for a segment.
            emg_channels (list[int] | None): Indices of EMG channels within the frame.
            id (int): Subject/session identifier used in file names.
            dateString (str): Short date string (dd-mm) used in file naming.
            ind (int): Internal counter for segments recorded in this process.
        """
    def __init__(self, use_emg, use_eeg):
        """Initialize a recording session and connect to the device.

                Args:
                    use_emg (bool): Whether EMG channels are enabled.
                    use_eeg (bool): Whether EEG channels are enabled.
                """
        self.config = Config(use_emg, use_eeg)
        self.socket_handler = SocketHandler(self.config.IP_ADDRESS, self.config.TCP_PORT)
        self.socket_handler.connect()
        self.conf_string = None
        self.tot_num_byte = None
        self.tot_num_chan = None
        self.recording = False
        self.emg_channels = None
        self.start()
        self.id = 0
        self.dateString = datetime.today().strftime('%d-%m')
        self.make_directory()
        self.ind = 0

    def start(self):
        """Validate and send the start/configuration command to the device.

                Validates `DEVICE_EN`, `EMG`, and `MODE`, computes the packed
                configuration with `process_config`, and sends it over the socket.
                Populates: `conf_string`, `emg_channels`, `tot_num_chan`, `tot_num_byte`.
                """
        # Validate the contents of the configuration arrays
        validate_config(self.config.DEVICE_EN, 1, "Error, set DeviceEN values equal to 0 or 1")
        validate_config(self.config.EMG, 1, "Error, set EMG values equal to 0 or 1")
        validate_config(self.config.MODE, 3, "Error, set Mode values between to 0 and 3")

        # Process the configuration to get the configuration string and other required fields
        self.conf_string, conf_str_len, self.emg_channels, self.tot_num_chan, self.tot_num_byte, plotting_info = (
            process_config(self.config.DEVICE_EN, self.config.EMG, self.config.MODE, self.config.NUM_CHAN))

        # Send the configuration to syncstation
        start_command = self.conf_string[0:conf_str_len]
        packed_data = struct.pack('B' * conf_str_len, *start_command)
        data_sent = self.socket_handler.send(packed_data)
        print(f"Start Command Sent: {start_command}")



    def finish(self):
        """Send a stop command and close the socket.

                Mutates the configuration header to craft a stop command, sends it,
                and closes the TCP connection.
                """
        # Send the stop command to syncstation
        for i in range(18):
            self.conf_string[i] = 0
        self.conf_string[1] = calculate_crc8(self.conf_string, 1)
        stop_command = self.conf_string[0:1]
        packed_data = struct.pack('B' * len(stop_command), *stop_command)
        print("Stop Command Sent")
        data_sent = self.socket_handler.send(packed_data)
        self.socket_handler.close()

    def emg_recording(self, perform_time, rest_time, movement, rep):
        """Record one movement + following rest segment for EMG.

        Args:
            perform_time (float): Duration of the movement in seconds.
            rest_time (float): Rest duration immediately after the movement in seconds.
            movement (int): Movement label/index (used for file naming and labels).
            rep (int): Repetition number (used for file naming).
        """
        print(f"Recording for movement {movement} rep {rep} ({perform_time + rest_time} seconds)")
        self.record(True, rest_time, movement, perform_time=perform_time, rep=rep)
    def record_initial_rest(self, rest_time, movement, perform_time):
        """Record the initial rest period before a movement.

        Args:
            rest_time (float): Rest duration in seconds.
            movement (int): Movement label/index (used in labels and naming).
            perform_time (float): Planned movement duration (for label sizing).
        """
        print(f"Recording initial rest for movement {movement} ({rest_time} seconds)")
        self.record(False, rest_time, movement, perform_time=perform_time)

    def record(self, is_movement, rest_time, movement, perform_time=0, rep=None):
        """Record a single segment (movement or rest), align, decode, and save.

        Streams raw bytes for `rec_time`, aligns frames (EEG-aware if enabled),
        reshapes to frames, decodes to channel arrays, builds movement/rest labels,
        and saves EMG/EEG/counter channels according to config.

        Args:
            is_movement (bool): If True, records `perform_time + rest_time`; else only rest.
            rest_time (float): Rest duration in seconds.
            movement (int): Movement label/index.
            perform_time (float, optional): Movement duration in seconds. Defaults to 0.
            rep (int | None): Repetition index for naming when `is_movement=True`.

        Notes:
            - Uses `SAMPLE_TOLERANCE` to accept minor sample count drift.
            - Uses `simple_alignment` when EEG is not enabled; otherwise no offset trim here.
        """
        if is_movement: rec_time = perform_time + rest_time
        else: rec_time = rest_time


        total_samples = int(self.config.SAMPLE_FREQUENCY * rec_time)
        expected_bytes = self.tot_num_byte * total_samples
        data = np.zeros((self.tot_num_chan, int(self.config.SAMPLE_FREQUENCY * rec_time)))

        chan_ready = 0
        data_buffer = b""  # Buffer to store the received data

        chunk_size = self.tot_num_byte * 10
        start_time = time.time()
        self.recording = True

        while time.time() - start_time < rec_time:
            data_temp = self.socket_handler.receive(chunk_size)
            if not data_temp:
                break
            data_buffer += data_temp
        self.ind +=1
        self.recording = False
        print(f"Elapsed time for receiving data: {time.time() - start_time:.2f} seconds")
        print("Total bytes received:", len(data_buffer))
        sample_size = self.tot_num_byte
        remainder = len(data_buffer) % sample_size
        if remainder != 0:
            data_buffer = data_buffer[remainder:]
        if len(data_buffer) >= expected_bytes:
            data_buffer = data_buffer[-expected_bytes:]
        else:
            print("Warning: received less data than expected")

        temp_array = np.frombuffer(data_buffer, dtype=np.uint8)
        temp = np.reshape(temp_array, (-1, self.tot_num_byte)).T  # dynamic reshape

        num_samples = temp.shape[1]
        expected_samples = self.config.SAMPLE_FREQUENCY * rec_time

        if num_samples != expected_samples and expected_samples - num_samples < SAMPLE_TOLERANCE:
            data = np.zeros((self.tot_num_chan, num_samples))
            print(f"Allowed {num_samples} samples")

        data = process(self.config, temp, data, self.tot_num_byte, chan_ready)

        labels = np.array([movement] * int(perform_time * self.config.SAMPLE_FREQUENCY) + [0] * int(rest_time * self.config.SAMPLE_FREQUENCY))
        labels = labels if is_movement else np.array([0] * int(rest_time * self.config.SAMPLE_FREQUENCY))


        suffix = f"M{movement}R{rep}" if is_movement else f"M{movement}rest"
        exercise_group = "EA" if movement < 13 else "EB"

        if self.config.USE_EMG:
            self.save_channels(data[self.config.MUOVI_EMG_CHANNELS], labels, "emg", perform_time, exercise_group, suffix)

        if self.config.USE_EEG:
            self.save_channels(data[self.config.MUOVI_PLUS_EEG_CHANNELS], labels, "eeg", perform_time, exercise_group, suffix)

        if self.config.SAVE_COUNTERS and self.config.USE_EEG:
            self.save_channels(np.array([data[self.config.SYNCSTATION_COUNTER_CHANNEL],
                                         data[self.config.MUOVI_PLUS_COUNTER_CHANNEL]]), labels, "counters", perform_time, exercise_group,
                               suffix)

        gc.collect()
    def receive_and_ignore(self, duration, no_print=False):
        """Passively read and discard incoming bytes for a duration.

                Useful for flushing the socket or letting the device stream while
                the UI prepares, without recording.

                Args:
                    duration (float): Seconds to continue ignoring data.
                    no_print (bool): Suppress the default 'Ignoring' print if True.
                """
        if not no_print: print("Ignoring")
        end_time = time.time() + duration
        while self.recording:
            time.sleep(0.05)
        while time.time() < end_time:
            if not self.recording:
                data = self.socket_handler.receive(1024)
                if not data:
                    break

    def set_id(self, new_id):
        """Set the current subject/session identifier.

                Args:
                    new_id (int): Identifier to embed in output file paths.
                """
        self.id = new_id

    def make_directory(self):
        """Ensure the base data destination directory exists."""
        dir_path = self.config.DATA_DESTINATION_PATH

        # Check if it exists and is a directory
        if not os.path.isdir(dir_path):
            # Create the directory (and any missing parent directories)
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")
        else:
            print(f"Directory already exists: {dir_path}")

    def make_subject_directory(self, subject_id, exercise_set):
        """Create (if needed) the subject directory tree for an exercise set.

                Args:
                    subject_id (int | str): Subject identifier used in the folder structure.
                    exercise_set (str): Exercise set label (e.g., 'EA' or 'EB').
                """
        make_subject_directory(
            self.config.DATA_DESTINATION_PATH,
            subject_id,
            exercise_set,
            use_emg=self.config.USE_EMG,
            use_eeg=self.config.USE_EEG,
            save_counters=self.config.SAVE_COUNTERS
        )

    def save_channels(self, data, labels, type_string, perform_time, exercise_group, suffix):
        """Persist a set of channels plus labels to disk.

                Args:
                    data (np.ndarray): Channel-major array to save (shape: [n_channels, n_samples]).
                    labels (np.ndarray): 1D label array per-sample for movement/rest.
                    type_string (str): Type key ('emg', 'eeg', or 'counters').
                    perform_time (float): Movement duration used in naming/metadata.
                    exercise_group (str): Exercise set/group label, e.g., 'EA'/'EB'.
                    suffix (str): File suffix encoding movement/rep or rest segment.
                """
        save_channels(
            self.config.DATA_DESTINATION_PATH,
            self.id,
            type_string,
            exercise_group,
            perform_time,
            suffix,
            data,
            labels,
            save_h5=self.config.SAVE_H5,
            date_str=self.dateString
        )

    def get_record(self, rec_time):
        """Capture a raw segment for `rec_time` seconds and return decoded channels. Used to make sure there is nonzero data for each device.

               Streams bytes for the specified duration, aligns frames (EEG-aware if
               enabled), decodes into channel arrays, and returns the numeric data.

               Args:
                   rec_time (float): Segment duration in seconds.

               Returns:
                   np.ndarray: Array of shape [n_channels, n_samples] for the captured segment.
               """

        data_buffer = b""
        chunk_size = self.tot_num_byte * 10
        start_time = time.time()
        self.recording = True

        while time.time() - start_time < rec_time:
            data_temp = self.socket_handler.receive(chunk_size)
            if not data_temp:
                break
            data_buffer += data_temp

        total_samples = int(self.config.SAMPLE_FREQUENCY * rec_time)
        expected_bytes = self.tot_num_byte * total_samples
        data = np.zeros((self.tot_num_chan, int(self.config.SAMPLE_FREQUENCY * rec_time)))

        chan_ready = 0
        data_buffer = b""  # Buffer to store the received data

        chunk_size = self.tot_num_byte * 10
        start_time = time.time()
        self.recording = True

        while time.time() - start_time < rec_time:
            data_temp = self.socket_handler.receive(chunk_size)
            if not data_temp:
                break
            data_buffer += data_temp
        self.recording = False
        if not self.config.USE_EEG:
            offset = simple_alignment(data_buffer)
        else:
            offset = 0
        if offset != 0:
            data_buffer = data_buffer[:-offset]
        sample_size = self.tot_num_byte
        remainder = len(data_buffer) % sample_size
        if remainder != 0:
            data_buffer = data_buffer[remainder:]
        if len(data_buffer) >= expected_bytes:
            data_buffer = data_buffer[-expected_bytes:]
        else:
            print("Warning: received less data than expected")

        temp_array = np.frombuffer(data_buffer, dtype=np.uint8)
        temp = np.reshape(temp_array, (-1, self.tot_num_byte)).T  # dynamic reshape
        data = np.zeros((self.tot_num_chan, temp.shape[1]))
        data = process(self.config, temp, data, self.tot_num_byte, chan_ready)
        return data
