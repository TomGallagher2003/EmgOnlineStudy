"""Configuration definitions for synchronized EMG/EEG data collection.

Exposes the `Config` class, which centralizes hardware enable flags,
channel maps, sample rate, network settings, gain modes, file paths,
and other runtime parameters used by the recording pipeline.
"""

class Config:
    """Experiment/runtime configuration for EMG/EEG acquisition.

    This class captures device enable flags, channel layouts, sample rate,
    IP/port for the SyncStation, gain mode settings, filesystem paths, and
    derived indices (e.g., which channels map to EMG, EEG, AUX, and counters).

    Attributes:
        USE_EMG (bool): Enable EMG device/channels.
        USE_EEG (bool): Enable EEG device/channels.
        SAVE_COUNTERS (bool): Save SyncStation/Muovi counter channels.
        SAVE_H5 (bool): Save HDF5 outputs in addition to CSV (if applicable).

        EMG_MODE (int): EMG gain mode (0 → gain 8, 1 → gain 4; 2/3 test).
        EEG_MODE (int): EEG gain mode (same encoding as EMG_MODE).
        GAIN_RATIOS (dict[int, float]): Mode→volts-per-count scale factors.

        DEVICE_EN (list[int]): Per-device enable mask (1=on, 0=off);
            EMG at index 0, EEG at index 4 (others reserved).
        EMG (list[int]): Per-channel enable mask for EMG device group.
        MODE (list[int]): Per-device mode values; EMG_MODE at index 0,
            EEG_MODE at index 4 (others reserved).

        TCP_PORT (int): SyncStation TCP port.
        IP_ADDRESS (str): SyncStation IP address.
        SAMPLE_FREQUENCY (int): Sampling rate in Hz for EMG/EEG streams.
        OFFSET_EMG (int): Optional EMG DC offset correction (counts).
        PLOT_TIME (int): Default window length for plotting (seconds).

        DATA_DESTINATION_PATH (str): Directory for signal outputs.
        LABEL_DESTINATION_PATH (str): Directory for label files.
        IMAGE_SOURCE_PATH (str): Directory for movement cue images.

        NUM_CHAN (list[int]): Bytes-per-device channel counts in stream order.

        MUOVI_EMG_CHANNELS (list[int]): Index range of EMG channels.
        MUOVI_AUX_CHANNELS (list[int]): Index range of EMG AUX channels.
        MUOVI_PLUS_EEG_CHANNELS (list[int]): Index range of EEG channels.
        MUOVI_PLUS_AUX_CHANNELS (list[int]): Index range of EEG AUX channels.
        SYNCSTATION_CHANNELS (list[int]): Index range of SyncStation channels.

        SYNCSTATION_COUNTER_CHANNEL (int): SyncStation counter channel index.
        MUOVI_COUNTER_CHANNEL (int): EMG-device counter channel index.
        MUOVI_PLUS_COUNTER_CHANNEL (int): EEG-device counter channel index.
    """


    def __init__(self, use_emg, use_eeg):
        """Initialize configuration with chosen devices and derive channel maps.

        Args:
            use_emg (bool): Whether to enable EMG acquisition.
            use_eeg (bool): Whether to enable EEG acquisition.

        Notes:
            - `DEVICE_EN`, `EMG`, and `MODE` arrays are structured for the
              underlying firmware protocol: EMG settings are at index 0,
              EEG settings at index 4. Other indices are reserved.
            - Channel index lists (e.g., `MUOVI_EMG_CHANNELS`) are computed
              in stream order, expanding as devices are enabled so that
              EMG, EEG, AUX, and SyncStation ranges are contiguous and
              consistent with the decoder.
        """

        self.USE_EMG = use_emg
        self.USE_EEG = use_eeg
        self.SAVE_COUNTERS = True
        self.SAVE_H5 = True

        # Set the Gain Mode here : 0 -> 8, 1 -> 4
        self.EMG_MODE = 0
        self.EEG_MODE = 0
        self.GAIN_RATIOS = {
            0: 286.1e-9,   # MODE=00 → 286.1 nV
            1: 572.2e-9,   # MODE=01 → 572.2 nV
            2: 1.0e-3,     # TEST MODE
            3: 1.0e-3      # TEST MODE
            }




        eeg_on = 1 if self.USE_EEG else 0
        emg_on = 1 if self.USE_EMG else 0
        self.DEVICE_EN = [emg_on, 0, 0, 0, eeg_on, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.EMG = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.MODE = [self.EMG_MODE, 0, 0, 0, self.EEG_MODE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


        # Configuration parameters
        self.TCP_PORT = 54320
        self.OFFSET_EMG = 1000
        self.PLOT_TIME = 1
        self.IP_ADDRESS = '192.168.76.1'



        self.NUM_CHAN = [38, 38, 38, 38, 70, 70, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
        self.SAMPLE_FREQUENCY = 2000
        self.DATA_DESTINATION_PATH = "./data"
        self.LABEL_DESTINATION_PATH = "./labels"
        self.IMAGE_SOURCE_PATH = "./movement_library/EA"


        num_channels_used = 0
        self.MUOVI_EMG_CHANNELS = list(range(0, 32))
        self.MUOVI_AUX_CHANNELS = list(range(32, 38))
        if self.USE_EMG:
            num_channels_used += 38

        self.MUOVI_PLUS_EEG_CHANNELS = list(range(num_channels_used, num_channels_used + 64))
        self.MUOVI_PLUS_AUX_CHANNELS = list(range(num_channels_used + 64, num_channels_used + 70))
        if self.USE_EEG:
            num_channels_used += 70

        self.SYNCSTATION_CHANNELS = list(range(num_channels_used, num_channels_used + 6))

        self.SYNCSTATION_COUNTER_CHANNEL = self.SYNCSTATION_CHANNELS[4]
        self.MUOVI_COUNTER_CHANNEL = self.MUOVI_AUX_CHANNELS[4]
        self.MUOVI_PLUS_COUNTER_CHANNEL = self.MUOVI_PLUS_AUX_CHANNELS[5]

