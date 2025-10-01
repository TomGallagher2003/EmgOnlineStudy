from util.filter_helpers import bandpass_filter, highpass_filter, lowpass_filter, notch_filter


def selective_filter(filters, data):
    """
    filters: list of dicts like
       {"type": "None"|"Pass"|"Notch", "lower": float|None, "upper": float|None, "center": float|None}
    Apply in order: A then B then C.
    - Pass:
        * lower only -> high-pass
        * upper only -> low-pass
        * both -> band-pass (lower < upper guaranteed by UI)
    - Notch:
        * center required
    - None: skip
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