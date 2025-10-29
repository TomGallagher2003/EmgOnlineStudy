---
title: Extending Classification Output
summary: A guide for adding more fields to the output of classification in the UI
---

# Extending Classification Output

---

## Baseline (what exists now)

- **Worker**: `ClassificationWorker` emits a minimal result like:
  ```python
  finished_ok.emit((pred_class, confidence))
  ```
- **UI**: `ExperimentPage._on_classification_done(result)` expects a 2‑tuple and updates `self.results_label`.
- **Artifacts**: A simple `classification_report.txt` is written.

---



## Step 1 — Define a result schema

**Option A: `dataclass` (recommended)**

```python
# workers/classification_types.py
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Union

@dataclass
class ClassificationResult:
    pred_class: int
    ... a bunch of new fields
```

*(Alternative: a `namedtuple` if you prefer.)*

---

## Step 2 — Produce output in `ClassificationWorker`

Inside `ClassificationWorker.run()` after inference, emit the new structure:

```python
# workers/classification.py


result = ClassificationResult(
    pred_class=pred_class,
  ... a bunch of new fields
)

# Emit ONLY the new object (preferred), or both for compatibility:
self.finished_ok.emit(result)                 # new rich object
# self.finished_ok.emit((pred_class, confidence))  # legacy, if needed
```

---

## Step 3 — Update the UI handler (backward compatible)

```python
# pages/experiment_page.py (or wherever your handler lives)
def _on_classification_done(self, result):
    self.is_classifying = False
    try:
        from workers.classification_types import ClassificationResult
        if isinstance(result, ClassificationResult):
            true_id = self.params.get("movement_id", "N/A")
            lines = [
                f"True: {true_id}",
                f"Pred: {result.pred_class} ({result.confidence*100:.1f}%)",
                ... new fields
            ]
            self.results_label.setText("  • " + "\n  • ".join(lines))
        else:
            # Legacy (pred_class, confidence)
            pred, conf = result
            if conf <= 1.0: conf *= 100.0
            true_id = self.params.get("movement_id", "N/A")
            self.results_label.setText(
                f"True {true_id} | Pred: {pred} at {conf:.1f}%"
            )
    except Exception:
        self.results_label.setText("Classification complete.")
    finally:
        self.status_label.setText("Classification complete.")
        self.btn_random.setEnabled(True)
        self.btn_start.setEnabled(self.session is not None and self.current_movement is not None)
```

---



## Step 5 — UI niceties (optional)

To show tables or graphs, look into pygpt widgets, suchs as pyqtgraph


---
## Step 6 — Minimal checklist

- [ ] Add `workers/classification_types.py` (dataclass).
- [ ] Build `ClassificationResult` in `ClassificationWorker.run()` and `emit` it.
- [ ] Update `_on_classification_done` to accept either tuple or dataclass.

