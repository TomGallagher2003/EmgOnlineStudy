from util.filter_helpers import bandpass_filter, highpass_filter, lowpass_filter, notch_filter

"""Composable signal filtering utilities.

This module exposes :func:`selective_filter`, a tiny dispatcher that applies up
to three user-specified filters in sequence (A → B → C). It’s designed to work
with the parameter dicts produced by your GUI (``ParametersPage.FilterRow``).
"""


def selective_filter(filters, data):
    """Apply a sequence of user-selected filters to an array.

    The function expects a list of filter-spec dictionaries—each with keys
    ``type``, ``lower``, ``upper``, and ``center``—and applies them in order.
    Supported modes:

    - **"Pass"**:
        - lower **and** upper → band-pass
        - lower only → high-pass
        - upper only → low-pass
    - **"Notch"**:
        - center required → notch at ``center`` Hz
    - **"None"**: do nothing

    Args:
        filters:
            Iterable of dicts like:
            ``{"type": "None"|"Pass"|"Notch", "lower": float|None, "upper": float|None, "center": float|None}``.
            They are applied **in order** (A then B then C).
        data:
            Signal array. Shape is project-specific (e.g., ``(samples, channels)``
            or ``(channels, samples)``). The helper filter functions must accept
            this shape.

    Returns:
        The filtered array (same shape as ``data``).

    Notes:
        - The UI guarantees ``lower < upper`` for band-pass. If both are ``None``
          for "Pass", this step is skipped.
        - This function does not inject a sampling rate; the imported helpers
          (``bandpass_filter``, etc.) must use project defaults or internal
          configuration.
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
    return out
