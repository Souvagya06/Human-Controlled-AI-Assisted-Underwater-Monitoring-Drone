"""
models/__init__.py
==================
Registers all custom YOLO-DarkWater modules into the Ultralytics task parser.

Patching strategy (based on Ultralytics 8.4.x parse_model source):
- GhostConv  : Ultralytics built-in — already in base_modules frozenset. No patch needed.
- GhostC2f   : Added to base_modules AND repeat_modules (identical channel logic to C2f).
               Then replaced in the built nn.Sequential with the actual GhostC2f instance.
- CBAM       : Unknown module → falls to 'else: c2 = ch[f]' (passthrough). Correct. ✓
               Instantiated as CBAM(*args) = CBAM(256) → c1=256, works perfectly.

Transfer learning:
- load_pretrained_partial() transfers shape-compatible weights from YOLOv8n.
- GhostConv / CBAM layers that differ in shape are skipped → keep random init.
"""

from __future__ import annotations

import torch  # required for "torch.nn.Module" type annotation in load_pretrained_partial

# ---------------------------------------------------------------------------
# Import custom modules
# ---------------------------------------------------------------------------
from models.ghost_conv import GhostBottleneck, GhostC2f
from models.attention import CBAM, ChannelAttention, SpatialAttention

# Use Ultralytics' built-in GhostConv (already in parse_model's base_modules)
from ultralytics.nn.modules import GhostConv

# ---------------------------------------------------------------------------
# Register custom modules in ultralytics.nn.tasks namespace
# (globals()[m] in parse_model resolves names from this namespace)
# ---------------------------------------------------------------------------
import ultralytics.nn.tasks as _tasks

_tasks.GhostC2f = GhostC2f  # type: ignore[attr-defined]
_tasks.CBAM = CBAM  # type: ignore[attr-defined]
# Patch parse_model to handle GhostC2f channel tracking
# ---------------------------------------------------------------------------
if not hasattr(_tasks, "_true_original_parse_model"):
    _tasks._true_original_parse_model = _tasks.parse_model  # type: ignore[attr-defined]

_original_parse_model = _tasks._true_original_parse_model  # type: ignore[attr-defined]
from ultralytics.nn.modules import C2f

_in_parse_model = False


def _darkwater_parse_model(d, ch, verbose=True):
    """
    Patched parse_model that adds proper channel tracking for GhostC2f and scales CBAM.
    Includes a recursion guard to prevent double-entry and duplicate scaling.
    """
    global _in_parse_model

    # If already parsing, delegate directly to original function to avoid infinite recursion
    if _in_parse_model:
        return _original_parse_model(d, ch, verbose)

    _in_parse_model = True
    try:
        from copy import deepcopy
        import torch
        from ultralytics.nn.tasks import make_divisible

        # Determine width multiple for scaling custom layers
        scale = d.get("scale")
        scales = d.get("scales")
        if scales:
            if not scale:
                # Fallback to the first scale key if scale is not specified
                scale = next(iter(scales.keys()))
            _, width, _ = scales[scale]
        else:
            width = d.get("width_multiple", 1.0)

        # ------------------------------------------------------------------
        # Step 1: Replace GhostC2f → C2f, scale CBAM in a YAML copy, track positions
        # ------------------------------------------------------------------
        d_proxy = deepcopy(d)
        ghost_positions = []  # [(abs_layer_idx, orig_args)]

        backbone = d_proxy.get("backbone", [])
        head = d_proxy.get("head", [])

        for i, layer in enumerate(backbone):
            if layer[2] == "GhostC2f":
                ghost_positions.append((i, layer[:]))
                d_proxy["backbone"][i][2] = "C2f"
            elif layer[2] == "CBAM":
                # Scale the channel argument (first element of args list)
                c2_raw = layer[3][0]
                c2_scaled = make_divisible(c2_raw * width, 8)
                d_proxy["backbone"][i][3][0] = c2_scaled

        backbone_len = len(backbone)
        for i, layer in enumerate(head):
            if layer[2] == "GhostC2f":
                ghost_positions.append((backbone_len + i, layer[:]))
                d_proxy["head"][i][2] = "C2f"
            elif layer[2] == "CBAM":
                c2_raw = layer[3][0]
                c2_scaled = make_divisible(c2_raw * width, 8)
                d_proxy["head"][i][3][0] = c2_scaled

        # ------------------------------------------------------------------
        # Step 2: Run original parse_model (GhostC2f now seen as C2f)
        # ------------------------------------------------------------------
        model_seq, save = _original_parse_model(d_proxy, ch, verbose)

        # ------------------------------------------------------------------
        # Step 3: Replace C2f → GhostC2f at tracked positions
        # ------------------------------------------------------------------
        for abs_idx, _ in ghost_positions:
            c2f = model_seq[abs_idx]

            # Extract parameters from the built C2f
            try:
                c1 = c2f.cv1.conv.in_channels  # type: ignore[union-attr]
                c2_out = c2f.cv2.conv.out_channels  # type: ignore[union-attr]
                n_blocks = len(c2f.m)  # type: ignore[arg-type]
                shortcut = bool(c2f.m[0].add) if n_blocks > 0 and hasattr(c2f.m[0], "add") else False  # type: ignore[index]
            except AttributeError:
                inner = c2f[0] if isinstance(c2f, torch.nn.Sequential) else c2f  # type: ignore[index]
                c1 = inner.cv1.conv.in_channels  # type: ignore[union-attr]
                c2_out = inner.cv2.conv.out_channels  # type: ignore[union-attr]
                n_blocks = len(inner.m)  # type: ignore[arg-type]
                shortcut = bool(inner.m[0].add) if n_blocks > 0 and hasattr(inner.m[0], "add") else False  # type: ignore[index]

            # Build GhostC2f with identical dimensions
            ghost_module = GhostC2f(c1, c2_out, n=n_blocks, shortcut=shortcut)

            # Transfer metadata attributes set by parse_model
            ghost_module.i = c2f.i  # type: ignore[attr-defined]
            ghost_module.f = c2f.f  # type: ignore[attr-defined]
            ghost_module.type = str(type(ghost_module))[8:-2].replace("__main__.", "")  # type: ignore[assignment]
            ghost_module.np = sum(x.numel() for x in ghost_module.parameters())  # type: ignore[assignment]

            # Replace in nn.Sequential's ordered module dict
            model_seq._modules[str(abs_idx)] = ghost_module

            if verbose:
                print(
                    f"  [DarkWater] Replaced C2f → GhostC2f at layer {abs_idx}"
                    f"  (c1={c1}, c2={c2_out}, n={n_blocks}, shortcut={shortcut})"
                )

        return model_seq, save
    finally:
        _in_parse_model = False


# Apply the patch
_tasks.parse_model = _darkwater_parse_model


# ---------------------------------------------------------------------------
# Transfer learning utility
# ---------------------------------------------------------------------------

def load_pretrained_partial(
    model: "torch.nn.Module",
    pretrained_path: str = "yolov8n.pt",
    verbose: bool = True,
) -> tuple[list[str], list[str]]:
    """
    Transfer compatible pretrained YOLOv8n weights into YOLO-DarkWater.

    Rules:
    - Only layers where the parameter name AND tensor shape both match are transferred.
    - Layers with shape mismatches (GhostConv, CBAM, or resized layers) keep their
      random initialization.
    - Uses strict=False to allow partial loading.

    Args:
        model:           The YOLO-DarkWater model (nn.Module).
        pretrained_path: Path to YOLOv8n .pt file.
        verbose:         Print transfer summary.

    Returns:
        transferred: List of parameter names successfully transferred.
        skipped:     List of parameter names skipped (shape mismatch or not found).
    """
    # Load pretrained weights (Ultralytics .pt saves the full model)
    ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)

    # Extract state dict from Ultralytics checkpoint format
    if isinstance(ckpt, dict):
        pretrained_state = ckpt.get("model", ckpt)
        if hasattr(pretrained_state, "state_dict"):
            pretrained_state = pretrained_state.state_dict()
        elif isinstance(pretrained_state, dict) and "model" in pretrained_state:
            pretrained_state = pretrained_state["model"].state_dict()
    else:
        pretrained_state = ckpt.state_dict()

    model_state = model.state_dict()
    new_state = model_state.copy()

    transferred = []
    skipped_shape = []
    skipped_missing = []

    for key, pretrained_param in pretrained_state.items():
        if key in model_state:
            if model_state[key].shape == pretrained_param.shape:
                new_state[key] = pretrained_param
                transferred.append(key)
            else:
                skipped_shape.append(f"{key} "
                                     f"(pretrained: {tuple(pretrained_param.shape)}, "
                                     f"model: {tuple(model_state[key].shape)})")
        else:
            skipped_missing.append(key)

    model.load_state_dict(new_state, strict=False)

    if verbose:
        print(f"\n[Transfer Learning] YOLOv8n → YOLO-DarkWater")
        print(f"  ✓ Transferred : {len(transferred)} / {len(model_state)} parameters")
        print(f"  ✗ Shape mismatch: {len(skipped_shape)} layers (random init kept)")
        print(f"  ✗ Not found   : {len(skipped_missing)} layers")
        if skipped_shape:
            print("\n  Shape mismatches (GhostConv/CBAM layers expected):")
            for s in skipped_shape[:10]:
                print(f"    {s}")
            if len(skipped_shape) > 10:
                print(f"    ... and {len(skipped_shape) - 10} more")

    skipped = [s.split(" ")[0] for s in skipped_shape] + skipped_missing
    return transferred, skipped


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "GhostConv",
    "GhostBottleneck",
    "GhostC2f",
    "CBAM",
    "ChannelAttention",
    "SpatialAttention",
    "load_pretrained_partial",
]
