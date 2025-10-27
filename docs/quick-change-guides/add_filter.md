---
title: Add a filter
summary: Add a new filter and wire it in to the UI
---

# Add a New Filter

This guide shows **exactly three edits** to add a new filter type end‑to‑end:
1) Implement the filter in `util/filter_helpers.py`
2) Wire it into `pipeline_sections/filters.py` (`selective_filter` dispatcher)
3) Add it to options in the UI (`pages/parameters.py`)

> ⚠️ **Do not remove or reorder any existing imports** (matches current project rule).  
> We’ll use **Bandstop** as the example.

---

## 1) Implement the filter core
**File:** `util/filter_helpers.py`

Add a function that mirrors the existing helpers (shape‑in/shape‑out stays the same).

```python
# util/filter_helpers.py

def bandstop_filter(x, low: float, high: float):
    """Apply a band-stop filter between `low` and `high` Hz.
    
    Args:
        x: Signal array (shape as used elsewhere in your project).
        low: Lower cutoff in Hz.
        high: Upper cutoff in Hz.
    Returns:
        Filtered array (same shape as `x`).
    """
    # TODO: implement the DSP (Butterworth/IIR using your lib of choice)
    # Keep the same shape convention as bandpass/lowpass/highpass.
    return x
```

**Tips**
- Keep the function name distinct (`bandstop_filter`) and signature similar to others.
- If your helpers expect `(samples, channels)` or `(channels, samples)`, **honor the same convention**.

---

## 2) Add the new type to `selective_filter`
**File:** `pipeline_sections/filters.py`

1. **Import** your function at the top **without removing other imports**:
```python
from util.filter_helpers import bandpass_filter, highpass_filter, lowpass_filter, notch_filter, bandstop_filter
```

2. **Handle** the new type in the dispatcher:
```python
def selective_filter(filters, data):
    """
    filters: list of dicts like
       {"type": "None"|"Pass"|"Notch"|"Bandstop", "lower": float|None, "upper": float|None, "center": float|None}
    Apply in order: A then B then C.
    """
    out = data
    for f in filters:
        t = f["type"]
        if t == "Pass":
            lo, up = f["lower"], f["upper"]
            if lo is not None and up is not None:
                out = bandpass_filter(out, lo, up)
            elif lo is not None:
                out = highpass_filter(out, lo)
            elif up is not None:
                out = lowpass_filter(out, up)
        elif t == "Notch":
            if f["center"] is not None:
                out = notch_filter(out, f["center"])
        # --- NEW: Bandstop ---
        elif t == "Bandstop":
            lo, up = f["lower"], f["upper"]
            if lo is not None and up is not None:
                out = bandstop_filter(out, lo, up)
        # ----------------------
    return out
```

**Notes**
- The UI already guarantees `lower < upper` for pass bands. Keep the same assumption for bandstop.
- If **either** bound is missing, simply skip the bandstop step (as above).

---

## 3) Add it to the UI
**File:** `pages/parameters.py`

1. **Add the option label**:
```python
class ParametersPage(QtWidgets.QWidget):
    FILTER_OPTIONS = ["None", "Pass", "Notch", "Bandstop"]
```

2. **Support values/validation** inside `FilterRow` (reusing the Lower/Upper inputs):
```python
class FilterRow(QtWidgets.QWidget):
    def value(self) -> dict:
        # ...
        if t == "Pass":
            out["lower"], out["upper"] = f(self.lower_edit), f(self.upper_edit)
        elif t == "Notch":
            out["center"] = f(self.center_edit)
        elif t == "Bandstop":  # NEW
            out["lower"], out["upper"] = f(self.lower_edit), f(self.upper_edit)
        return out

    def validate(self, parent: QtWidgets.QWidget) -> bool:
        # ...
        if t == "Bandstop":  # NEW
            lower = self.lower_edit.text().strip()
            upper = self.upper_edit.text().strip()
            if not lower or not upper:
                QtWidgets.QMessageBox.warning(parent, "Invalid Bandstop filter",
                    "Lower and Upper Hz are required for a Bandstop filter.")
                return False
            if float(lower) >= float(upper):
                QtWidgets.QMessageBox.warning(parent, "Invalid Bandstop band",
                    "Lower Hz must be strictly less than Upper Hz.")
                return False
        return True
```


---

## Sanity checklist

- [ ] `bandstop_filter` implemented in `util/filter_helpers.py`
- [ ] Imported in `pipeline_sections/filters.py` and handled in `selective_filter`
- [ ] Added to `FILTER_OPTIONS` and supported in `FilterRow.value/validate`
- [ ] (Optional) Defaults wired into the EMG defaults button
