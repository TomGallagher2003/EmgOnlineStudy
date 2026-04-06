DATA_DESTINATION_PATH = "./data"

# Model Info
MODEL_PATH = str("./pipeline_sections/models/model.pth")
EEG_MODEL_PATH = str("./pipeline_sections/models/eeg_model.pth")
FUSION_MODEL_PATH = str("./pipeline_sections/models/fusion_model.pth")

# MODEL_MODE: "emg" | "eeg" | "fusion"
MODEL_MODE = "emg"

MODEL_IS_EEG = False             # False: EMG, True: EEG (legacy, derived from MODEL_MODE)
CLASSIFY_PROCESSED_DATA = False  # False: raw data used for classification, True: processed data used for classification

