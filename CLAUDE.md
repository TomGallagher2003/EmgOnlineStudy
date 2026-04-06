# CLAUDE.md — EmgOnlineStudy

## Token Efficiency Rules (MANDATORY)

- **No preamble.** Never restate the task, say "Great!", or describe what you're about to do.
- **No trailing summaries.** Don't recap what you just did at the end of responses.
- **Lead with the action or answer.** Text output only when it adds information.
- **No unsolicited explanations.** Explain only if asked or if something is non-obvious.
- **No filler.** Cut "I'll now...", "Let me...", "Sure, I can...", "Here's...".
- **One-sentence max** for status updates at milestones.
- **No docstrings, comments, or type annotations** on code you didn't change.
- **No extra features**, error handling, or abstractions beyond what was asked.
- **No backwards-compatibility hacks** — delete unused code entirely.
- **Verify before recommending** — grep for functions/files before citing them.

## Project: EmgOnlineStudy

Real-time EMG/EEG capture and classification system (PyQt5 + PyTorch).

### Stack
- Python 3.13, PyQt5 5.15.11, NumPy ~2.1.3, SciPy 1.15.2, PyTorch, h5py, Matplotlib
- Hardware: SyncStation/Muovi devices via TCP (192.168.76.1:54320)
- Docs: MkDocs Material + mkdocstrings

### Entry Points
- `main.py` → QApplication → `pages/main.py` (MainWindow/stacked widget)
- Three-page flow: DeviceSelect → Parameters → Experiment

### Key Directories
- `pages/` — PyQt5 UI pages (device_select, parameters, experiment, main controller)
- `workers/` — QThread workers (recording, pipeline, classification, device_init, flush)
- `pipeline_sections/` — Signal processing (filters, normalisation, windows, models/)
- `util/` — Low-level utilities (recording session, socket, processing, file_pathing, images)
- `widgets/` — Custom PyQt5 widgets (arc_timer)
- `movement_library/` — PNG movement cue images (EA/: 12 movements, EB/: 18 movements)
- `docs/` — MkDocs documentation

### Key Files
| File | Role |
|------|------|
| `config.py` | Device flags, channels, sample rates (2000Hz EMG, 500Hz EEG), TCP endpoint |
| `main_settings.py` | Model path, MODEL_IS_EEG, CLASSIFY_PROCESSED_DATA flags |
| `emg_default_settings.py` | Window (256ms), overlap, filter defaults |
| `util/recording.py` | Session class: TCP comms, frame alignment, channel extraction |
| `util/processing.py` | Raw bytes → mV (EMG 16-bit, EEG 24-bit) |
| `util/socket_handling.py` | TCP socket wrapper with retry/flush |
| `util/file_pathing.py` | Directory creation, CSV/HDF5 saving |
| `util/images.py` | Movement registry (30 cues + rest) |
| `pipeline_sections/filters.py` | Filter dispatcher (bandpass/highpass/lowpass/notch) |
| `pipeline_sections/models/full_training.py` | CNN1D, TransformerModel, CNN1D_Transformer |
| `pipeline_sections/classify.py` | Evaluation wrapper: HDF5 → inference → predictions |
| `workers/recording.py` | RecordingWorker (fixed-duration capture) |
| `workers/pipeline.py` | PipelineWorker (filter→normalise→window off-thread) |
| `workers/classification.py` | ClassificationWorker (model inference) |

### Architecture Patterns
- **QThread workers** for all I/O and compute; UI stays responsive
- Workers emit `finished_ok(result)` / `failed(msg)` signals
- **No direct UI modification from worker threads**
- Modular pipeline: filter → normalise → window → inference
- HDF5 for structured data (`emg_data`, `emg_label` datasets)

### Data Output Structure
```
./data/trial_<N>/rec_<M>/
  raw_emg.csv, raw_emg.h5, processed_emg.h5
  raw_eeg.csv, processed_eeg.h5  (if EEG enabled)
  label.csv, classification_report.txt
```

### Channel Map
- EMG: channels 0–31 (main), 32–37 (aux)
- EEG: channels 38–101 (main), 102–107 (aux)
- SyncStation counters: 108–113

### Signal Processing Chain
Raw bytes → decode (EMG 16-bit / EEG 24-bit) → scale (286.1 nV/count) → filter → window (256ms) → normalise (min-max) → inference

### Movement Sets
- EA: 12 single-finger movements (flexion/extension + thumb variants)
- EB: 18 multi-finger & wrist movements
- Rest cue: `movement_library/Rest_M0.png`
