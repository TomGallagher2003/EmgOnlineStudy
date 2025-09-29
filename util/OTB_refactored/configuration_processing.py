
""" A collection of refactored functions containing the code provided in Syncstation.py from OT Bioelettronica"""
import os
import re


def calculate_crc8(vector, length):
    """Function to calculate CRC8"""

    crc = 0
    j = 0

    while length > 0:
        extract = vector[j]
        for i in range(8, 0, -1):
            Sum = crc % 2 ^ extract % 2
            crc //= 2

            if Sum > 0:
                str_crc = []
                a = format(crc, '08b')
                b = format(140, '08b')
                for k in range(8):
                    str_crc.append(int(a[k] != b[k]))

                crc = int(''.join(map(str, str_crc)), 2)

            extract //= 2

        length -= 1
        j += 1

    return crc


def validate_config(values, max_value, error_message):
    """ Refactored function to validate config values as in the orginal code """

    if any(value > max_value for value in values):
        print(error_message)
        exit()


def process_config(DeviceEN, EMG, Mode, NumChan):
    """ Processes the configurations to return: Configuration String, Configuration String Length"""

    size_comm = sum(DeviceEN)

    num_emg_chan_muovi, num_aux_chan_muovi = 0, 0
    num_emg_chan_sessn, num_aux_chan_sessn = 0, 0
    num_emg_chan_due_pl, num_aux_chan_due_pl = 0, 0

    muovi_emg_chan, muovi_aux_chan = [], []
    sessn_emg_chan, sessn_aux_chan = [], []
    due_pl_emg_chan, due_pl_aux_chan = [], []

    tot_num_chan = 0
    tot_num_byte = 0
    conf_str_len = 1
    conf_string = [0] * 18

    conf_string[0] = size_comm * 2 + 1

    for i in range(16):
        if DeviceEN[i] == 1:
            conf_string[conf_str_len] = (i * 16) + EMG[i] * 8 + Mode[i] * 2 + 1

            if i < 5:
                muovi_emg_chan.extend(list(range(tot_num_chan + 1, tot_num_chan + 33)))
                muovi_aux_chan.extend(list(range(tot_num_chan + 33, tot_num_chan + 39)))
                num_emg_chan_muovi += 32
                num_aux_chan_muovi += 6
            elif i > 6:
                due_pl_emg_chan.extend(list(range(tot_num_chan + 1, tot_num_chan + 3)))
                due_pl_aux_chan.extend(list(range(tot_num_chan + 3, tot_num_chan + 9)))
                num_emg_chan_due_pl += 2
                num_aux_chan_due_pl += 6
            else:
                sessn_emg_chan.extend(list(range(tot_num_chan + 1, tot_num_chan + 65)))
                sessn_aux_chan.extend(list(range(tot_num_chan + 65, tot_num_chan + 71)))
                num_emg_chan_sessn += 64
                num_aux_chan_sessn += 6

            tot_num_chan += NumChan[i]

            if EMG[i] == 1:
                tot_num_byte += NumChan[i] * 2
            else:
                tot_num_byte += NumChan[i] * 3


            conf_str_len += 1

    sync_stat_chan = list(range(tot_num_chan, tot_num_chan + 7))
    tot_num_chan += 6
    tot_num_byte += 12

    conf_string[conf_str_len] = 0  # Placeholder for CRC8 calculation

    # INITIALIZE PLOT
    plotting_info = PlottingInfo()

    if num_emg_chan_muovi != 0:
        plotting_info.set_mouvi_chans(muovi_emg_chan, muovi_aux_chan)

    if num_emg_chan_sessn == 0:
        plotting_info.set_sessn_chans(sessn_emg_chan, sessn_aux_chan)

    if num_emg_chan_due_pl != 0:
        plotting_info.set_due_pl_chans(due_pl_emg_chan, due_pl_aux_chan)

    # Calculate CRC8 and update conf_string
    conf_string[conf_str_len] = calculate_crc8(conf_string, conf_str_len)
    conf_str_len += 1
    return (conf_string, conf_str_len, muovi_emg_chan, tot_num_chan, tot_num_byte,
            plotting_info)


class PlottingInfo:

    def __init__(self):
        self.mouvi_emg_chan = None
        self.muovi_aux_chan = None
        self.due_pl_emg_chan = None
        self.due_pl_aux_chan = None
        self.sessn_emg_chan = None
        self.sessn_aux_chan = None

        self.mouvi_connected = False
        self.sessan_connected = False
        self.due_pl_connected = False

    def set_mouvi_chans(self, mouvi_emg_chan, muovi_aux_chan):
        self.mouvi_emg_chan = mouvi_emg_chan;
        self.muovi_aux_chan = muovi_aux_chan;
        self.mouvi_connected = True

    def set_sessn_chans(self, sessn_emg_chan, sessn_aux_chan):
        self.sessn_emg_chan = sessn_emg_chan;
        self.sessn_aux_chan = sessn_aux_chan;
        self.mouvi_connected = True

    def set_due_pl_chans(self, due_pl_emg_chan, due_pl_aux_chan):
        self.due_pl_emg_chan = due_pl_emg_chan;
        self.due_pl_aux_chan = due_pl_aux_chan;
        self.mouvi_connected = True


def get_sorted_file_paths(folder_path: str) -> list:
    """Get file paths and sort based on the M number in the filename."""
    file_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".png"):
                file_paths.append( file)

    # Sort files by the 'M' number in the filename
    file_paths.sort(key=lambda x: int(re.search(r'M(\d+)\.png', x, re.IGNORECASE).group(1)))
    return file_paths