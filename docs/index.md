# Immediate Classification EMG/EEG Study - Overview

This site documents the **Online EMG/EEG Study** app — a PyQt5 desktop application for
interactive data collection, on-the-fly processing, and quick EMG classification.

## Warning

Please don't remove any imports (even if your IDE marks them as unused). The "unused" imports are passed around to facilitate model loading, and their necessity isnt recognised by IDEs. This means issue will rise when using any built-in 'optimis imports' function offered by your IDE, so please avoid this.

## Module Overview


---

## Quick Start

1. **Install deps** (venv recommended):
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app**:
   ```bash
   python main.py
   ```

3. **Workflow**: *Device Select → Parameters → Experiment*
   - Pick EMG/EEG devices
   - Set window/overlap, filters, segmentation, recording length
   - Record a trial, process it, and classify EMG windows

    

## API Index

> Click into any of the following API pages for detailed references generated with mkdocstrings.

### Entry Points & Windows
- [`main`](api/main.md)
- 
### GUI Pages
- [`pages.device_select`](api/pages/device_select.md)
- [`pages.parameters`](api/pages/parameters.md)
- [`pages.experiment`](api/pages/experiment.md)

### Background Workers
- [`workers.device_init`](api/workers/device_init.md)
- [`workers.flush`](api/workers/flush.md)
- [`workers.recording`](api/workers/recording.md)
- [`workers.pipeline`](api/workers/pipeline.md)
- [`workers.classification`](api/workers/classification.md)

### Pipeline Sections
- [`pipeline_sections.filters`](api/pipeline_sections/filters.md)
- [`pipeline_sections.normalisation`](api/pipeline_sections/normalisation.md)
- [`pipeline_sections.reduce_eeg_samples`](api/pipeline_sections/reduce_eeg_samples.md)
- [`pipeline_sections.windows`](api/pipeline_sections/windows.md)

### Utilities & Widgets
- [`util.images`](api/util/images.md)
- [`util.mask_to_segments`](api/util/mask_to_segments.md)
- [`util.movement_segmentation`](api/util/movement_segmentation.md)
- [`util.recording`](api/util/recording.md)
- [`widgets.arc_timer`](api/widgets/arc_timer.md)

---
