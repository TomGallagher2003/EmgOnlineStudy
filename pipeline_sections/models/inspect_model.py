# summarize_state_dict.py
import argparse
import sys
import re
from collections import defaultdict, OrderedDict
from typing import Dict, Tuple, Any

import torch


def load_checkpoint(path: str, device: str = "cpu", unpickle: bool = False) -> Dict[str, torch.Tensor]:
    """
    Load a checkpoint. Default is safe (weights_only=True).
    If the file is a full model and you trust it, pass --unpickle to allow unpickling.
    Returns a state_dict-like dict mapping names -> tensors.
    """
    try:
        obj = torch.load(path, map_location=device, weights_only=True)
    except Exception as e:
        if not unpickle:
            raise RuntimeError(
                f"Failed safe load (weights_only=True). If this file is a full model "
                f"you trust, rerun with --unpickle.\nInner error: {e}"
            ) from e
        obj = torch.load(path, map_location=device, weights_only=False)

    if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
        return obj

    # Try extracting .state_dict() if it's a full model object
    if hasattr(obj, "state_dict") and callable(getattr(obj, "state_dict")):
        try:
            sd = obj.state_dict()
            if isinstance(sd, dict):
                return sd
        except Exception:
            pass

    raise TypeError(
        f"Loaded object is not a state_dict (type={type(obj)}). "
        f"Use --unpickle for full models or convert to a state_dict first."
    )


def strip_dataparallel_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove 'module.' prefix introduced by DataParallel, if present."""
    if not any(k.startswith("module.") for k in sd.keys()):
        return sd
    return {k[len("module."):]: v for k, v in sd.items()}


def group_prefix(key: str) -> str:
    """
    Return a short group prefix for display: up to the second dot if present,
    e.g., 'cnn1d.conv1', 'transformer.fc', '0.proj', '1.transformer'.
    """
    parts = key.split(".")
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return parts[0]


def human_bytes(n: int) -> str:
    # rough tensor memory footprint (assumes float32 unless dtype examined)
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def summarize(sd: Dict[str, torch.Tensor]) -> None:
    print("\n=== STATE_DICT SUMMARY ===")
    print(f"Total tensors: {len(sd)}")

    # Basic per-key info
    total_params = 0
    dtype_counts = defaultdict(int)
    group_params = defaultdict(int)

    # Try to infer special shapes
    adapter_info = {}      # in/out/k from '0.proj.weight' or '*.proj.weight'
    cnn_in_ch = None       # from cnn1d.conv1.weight[:, in_ch, :]
    d_model = None         # from self_attn.in_proj_weight shape = (3*d_model, d_model)
    fc_classes = None      # from transformer.fc.weight shape = (num_classes, fc_in)
    fc_in = None
    seen_attn = False

    # Prefer exact keys if present; otherwise fallback with regex
    adapter_weight_key = None
    if "0.proj.weight" in sd:
        adapter_weight_key = "0.proj.weight"
    else:
        for k in sd.keys():
            if k.endswith("proj.weight") and k.count(".") >= 2:
                adapter_weight_key = k
                break

    if adapter_weight_key:
        w = sd[adapter_weight_key]
        if w.ndim == 3:
            adapter_info = {
                "key": adapter_weight_key,
                "out_ch": int(w.shape[0]),
                "in_ch": int(w.shape[1]),
                "kernel": int(w.shape[2]),
            }

    # CNN conv1 input channels (after adapter out_ch if adapter exists)
    if "1.cnn1d.conv1.weight" in sd:
        w = sd["1.cnn1d.conv1.weight"]
        if w.ndim == 3:
            cnn_in_ch = int(w.shape[1])
    elif "cnn1d.conv1.weight" in sd:
        w = sd["cnn1d.conv1.weight"]
        if w.ndim == 3:
            cnn_in_ch = int(w.shape[1])

    # Transformer attention in_proj_weight → d_model
    attn_keys = [
        "1.transformer.transformer_layer.self_attn.in_proj_weight",
        "transformer.transformer_layer.self_attn.in_proj_weight",
        "1.transformer.transformer.layers.0.self_attn.in_proj_weight",
        "transformer.transformer.layers.0.self_attn.in_proj_weight",
    ]
    for k in attn_keys:
        if k in sd and sd[k].ndim == 2:
            W = sd[k]
            # shape: (3*d_model, d_model)
            d_model = int(W.shape[1])
            seen_attn = True
            break

    # FC head (num_classes)
    fc_keys = [
        "1.transformer.fc.weight",
        "transformer.fc.weight",
    ]
    for k in fc_keys:
        if k in sd and sd[k].ndim == 2:
            W = sd[k]
            fc_classes, fc_in = int(W.shape[0]), int(W.shape[1])
            break

    # Print inferred info first
    print("\n--- Inferred architecture bits ---")
    if adapter_info:
        print(f"Adapter weight:      {adapter_info['key']}: "
              f"[out_ch={adapter_info['out_ch']}, in_ch={adapter_info['in_ch']}, k={adapter_info['kernel']}]")
    else:
        print("Adapter weight:      (not found)")

    if cnn_in_ch is not None:
        print(f"CNN conv1 in_ch:     {cnn_in_ch}")
    else:
        print("CNN conv1 in_ch:     (not found)")

    if seen_attn and d_model is not None:
        print(f"Transformer d_model: {d_model}  (so valid num_heads must divide {d_model})")
    else:
        print("Transformer d_model: (not found)")

    if fc_classes is not None:
        print(f"Head (fc) classes:   {fc_classes}   | fc_in={fc_in}")
    else:
        print("Head (fc) classes:   (not found)")

    print("\n--- Per-parameter listing ---")
    # Sorted for stable output
    for name in sorted(sd.keys()):
        t = sd[name]
        shape = tuple(t.shape)
        dtype_counts[str(t.dtype)] += 1
        numel = t.numel()
        total_params += numel
        group_params[group_prefix(name)] += numel
        print(f"{name:70s}  shape={str(shape):20s}  dtype={str(t.dtype):10s}  numel={numel}")

    bytes_est = total_params * 4  # rough estimate (float32)
    print("\n--- Totals ---")
    print(f"Total parameters (numel): {total_params:,}")
    print(f"Approx memory (fp32):     {human_bytes(float(bytes_est))}")

    print("\n--- Dtype counts ---")
    for dt, cnt in sorted(dtype_counts.items(), key=lambda x: x[0]):
        print(f"{dt:>12s}: {cnt}")

    print("\n--- Parameter counts by group ---")
    for grp, cnt in sorted(group_params.items(), key=lambda x: x[0]):
        print(f"{grp:30s}: {cnt:,}")


def main():
    ap = argparse.ArgumentParser(description="Summarize a PyTorch state_dict (.pth) — shapes, dtypes, totals, and inferred bits.")
    ap.add_argument("path", help="Path to .pth file (state_dict preferred).")
    ap.add_argument("--device", default="cpu", help="map_location for torch.load (default: cpu)")
    ap.add_argument("--unpickle", action="store_true",
                    help="Allow unpickling (weights_only=False) if file is a full serialized model you trust.")
    args = ap.parse_args()

    sd = load_checkpoint(args.path, device=args.device, unpickle=args.unpickle)
    sd = strip_dataparallel_prefix(sd)
    summarize(sd)


if __name__ == "__main__":
    main()
