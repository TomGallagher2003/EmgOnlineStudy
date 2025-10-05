# classify_emg.py
from typing import Optional, Sequence, Tuple, Union, Dict, Any
import numpy as np
import torch
import torch.nn as nn

# Project modules
from pipeline_sections.models.evaluation import ChannelAdapter
from pipeline_sections.models.full_training import CNN1D_Transformer

TensorLike = Union[np.ndarray, torch.Tensor]


def _ensure_3d_nc_t(x: TensorLike) -> torch.Tensor:
    """Ensure float32 tensor shaped [N, C, T]; accepts [N, T, C] and permutes."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if x.ndim != 3:
        raise ValueError(f"Expected 3D windowed EMG [N,*,*], got {tuple(x.shape)}")
    _, A, B = x.shape
    if A >= B:  # [N, T, C] -> [N, C, T]
        x = x.permute(0, 2, 1)
    return x.contiguous().float()


def _majority_vote_confidence(preds_np: np.ndarray) -> Tuple[int, float]:
    counts = np.bincount(preds_np.astype(int))
    k = int(np.argmax(counts))
    conf = 100.0 * float(counts[k]) / float(len(preds_np))
    return k, round(conf, 2)


def _build_wrapped_model(
    in_ch: int,
    length_t: int,
    *,
    adapter_out_ch: int,
    num_classes: int,
    embed_dim: int = 128,
    num_heads: int = 8,
    num_layers: int = 3,
    adapter_k: int = 3,
    adapter_use_bn: bool = True,
) -> torch.nn.Module:
    """
    Training-time wrapper:
      nn.Sequential(ChannelAdapter(in_ch=?, out_ch=?, k=3, ...), CNN1D_Transformer(input_channels=out_ch, ...))
    """
    adapter = ChannelAdapter(in_ch=in_ch, out_ch=adapter_out_ch, k=adapter_k, causal=False, use_bn=adapter_use_bn)
    core = CNN1D_Transformer(
        input_channels=adapter_out_ch,
        length=length_t,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        num_classes=num_classes,
    )
    return nn.Sequential(adapter, core)


def _safe_load_obj(path: str, device: torch.device):
    """torch.load with weights_only=True; returns object (dict or full model depending on file)."""
    return torch.load(path, map_location=device, weights_only=True)


def _infer_ckpt_info(sd: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """
    Infer adapter and head sizes from a wrapped checkpoint (state_dict).
      - adapter weight at '0.proj.weight': [out_ch, in_ch, k]
      - head weight at '1.transformer.fc.weight': [num_classes, ...]
    """
    info: Dict[str, int] = {}
    w = sd.get("0.proj.weight")
    if isinstance(w, torch.Tensor) and w.ndim == 3:
        info["ckpt_adapter_out"] = int(w.shape[0])
        info["ckpt_adapter_in"]  = int(w.shape[1])
        info["ckpt_adapter_k"]   = int(w.shape[2])
    fc = sd.get("1.transformer.fc.weight")
    if isinstance(fc, torch.Tensor) and fc.ndim >= 2:
        info["ckpt_num_classes"] = int(fc.shape[0])
    return info


@torch.no_grad()
def classify_emg(
    windowed_emg: TensorLike,
    model_path: str = "pipeline_sections/models/model_state_dict.pth",
    device: Optional[str] = None,
    batch_size: int = 512,
    class_map: Optional[Sequence[str]] = None,
    arch_overrides: Optional[Dict[str, Any]] = None,
    trust_serialized_model: bool = False,
) -> Tuple[Union[int, str], float]:
    """
    Classify a batch of windowed, preprocessed EMG segments.

    - Accepts [N, T, C] or [N, C, T]; normalizes to [N, C, T].
    - Tries safe full-model load (weights_only=True, allowlist), then unpickle if trusted.
    - Fallback: load state_dict safely, infer adapter/class shapes, build wrapper accordingly, strict load.
    """
    batch_size = windowed_emg.shape[1]
    # Resolve device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Prepare input
    x = _ensure_3d_nc_t(windowed_emg)  # [N, C, T]
    if x.shape[0] == 0:
        raise ValueError("No windows provided for classification.")
    N, C, T = x.shape

    # Divisibility constraint: d_model = T//4 must be divisible by num_heads
    num_heads = (arch_overrides or {}).get("num_heads", 8)
    d_model = T // 4
    if (T % 4 != 0) or (d_model % num_heads != 0):
        req = 4 * num_heads
        raise ValueError(
            f"Incompatible window length T={T} for num_heads={num_heads}; "
            f"d_model=T//4={d_model} must be divisible by num_heads. Use T multiple of {req} (e.g., 512)."
        )

    model = None
    full_loaded = False

    # ---- Attempt 1: full serialized model with weights_only=True (allowlist)
    try:
        from torch.serialization import safe_globals
        with safe_globals([
            nn.Sequential, nn.ModuleList, nn.Identity, nn.Dropout, nn.LayerNorm,
            nn.Conv1d, nn.BatchNorm1d, nn.ReLU, nn.LeakyReLU, nn.MaxPool1d, nn.Linear,
            nn.TransformerEncoder, nn.TransformerEncoderLayer,
            ChannelAdapter,
        ]):
            model = torch.load(model_path, map_location=device, weights_only=True)
            model.eval()
            _ = model(x[:1].to(device))
            full_loaded = True
    except Exception:
        model = None
        full_loaded = False

    # ---- Attempt 2: unpickle if trusted
    if (not full_loaded) and trust_serialized_model:
        try:
            model = torch.load(model_path, map_location=device, weights_only=False)
            model.eval()
            _ = model(x[:1].to(device))
            full_loaded = True
        except Exception:
            model = None
            full_loaded = False

    # ---- Fallback: treat as state_dict (preferred long-term)
    if not full_loaded:
        obj = _safe_load_obj(model_path, device)
        if not isinstance(obj, dict):
            raise TypeError(
                f"Checkpoint at {model_path} is not a state_dict (got {type(obj).__name__}). "
                f"Use trust_serialized_model=True or convert it to a state_dict first."
            )
        sd: Dict[str, torch.Tensor] = obj

        # Infer adapter and classes from checkpoint
        info = _infer_ckpt_info(sd)
        ckpt_in  = info.get("ckpt_adapter_in")
        ckpt_out = info.get("ckpt_adapter_out")
        ckpt_k   = info.get("ckpt_adapter_k", 3)
        ckpt_classes = info.get("ckpt_num_classes", (arch_overrides or {}).get("num_classes", 18))

        # Hard-validate that runtime channels match the checkpoint adapter input (you said it's 32)
        if ckpt_in is not None and C != ckpt_in:
            raise ValueError(
                f"Channel mismatch: checkpoint adapter expects in_ch={ckpt_in}, but input has C={C}. "
                f"Ensure your EMG windows are shaped [N, {ckpt_in}, T]."
            )

        # Build wrapper EXACTLY to checkpoint: in_ch=C (== ckpt_in), out_ch=ckpt_out, num_classes=ckpt_classes
        # (CNN1D_Transformer will then take input_channels=ckpt_out)
        num_layers = (arch_overrides or {}).get("num_layers", 3)
        embed_dim  = (arch_overrides or {}).get("embed_dim", 128)

        model = _build_wrapped_model(
            in_ch=C,
            length_t=T,
            adapter_out_ch=ckpt_out if ckpt_out is not None else C,
            num_classes=ckpt_classes,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            adapter_k=ckpt_k,
        ).to(device)

        # Strict load should now succeed (shapes match)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[warn] load_state_dict mismatches — missing: {missing}, unexpected: {unexpected}")

    model.eval().to(device)

    # Batched inference
    preds = []
    for s in range(0, N, batch_size):
        xb = x[s:s + batch_size].to(device, non_blocking=True)
        logits = model(xb)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        pred = torch.argmax(logits, dim=1)
        preds.append(pred.cpu())
    preds = torch.cat(preds, dim=0).numpy()

    k, conf = _majority_vote_confidence(preds)
    label = class_map[k] if class_map is not None else k
    return label, conf


# Quick sanity test (optional)
if __name__ == "__main__":
    np.random.seed(42); torch.manual_seed(42)
    N, C, T = 8, 32, 512  # <-- 32 channels per your data
    dummy = np.random.randn(N, C, T).astype("float32")
    try:
        y, p = classify_emg(
            dummy,
            model_path="models/model_state_dict.pth",  # state_dict converted from your pickle
            batch_size=4,
        )
        print(f"Predicted: {y}, at {p:.2f}% confidence")
    except Exception as e:
        print(f"[demo] Classification failed: {e}")
