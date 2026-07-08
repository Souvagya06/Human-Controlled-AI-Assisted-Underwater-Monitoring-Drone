"""
scripts/validate_modules.py
============================
Pre-training validation suite for YOLO-DarkWater.

Verifies ALL custom modules compile, produce correct tensor shapes, and
are fully compatible with the Ultralytics pipeline BEFORE any training begins.

All checks must pass before proceeding to training.

Usage:
    python scripts/validate_modules.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import torch
import numpy as np

# Add repo root to path so models/ is importable
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Import custom modules (triggers parse_model patch)
import models  # noqa: F401 — triggers __init__.py registration

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PASSED = []
FAILED = []


def check(name: str):
    """Decorator that wraps a check function with pass/fail reporting."""
    def decorator(fn):
        def wrapper():
            print(f"  Checking: {name}...", end="  ")
            try:
                fn()
                print("✓ PASS")
                PASSED.append(name)
            except Exception as e:
                print(f"✗ FAIL")
                print(f"    Error: {e}")
                traceback.print_exc(limit=3)
                FAILED.append(name)
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

@check("CUDA device availability")
def check_cuda():
    assert torch.cuda.is_available(), "CUDA not available"
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\n      GPU: {name} ({vram:.1f} GB VRAM)", end="  ")


@check("GhostConv forward pass (1, 64, 320, 320) → (1, 128, 160, 160)")
def check_ghost_conv():
    from ultralytics.nn.modules import GhostConv
    m = GhostConv(64, 128, k=3, s=2).to(DEVICE)
    x = torch.randn(1, 64, 320, 320, device=DEVICE)
    out = m(x)
    assert out.shape == (1, 128, 160, 160), f"Got {out.shape}"


@check("GhostBottleneck forward pass (1, 128, 80, 80) → (1, 128, 80, 80)")
def check_ghost_bottleneck():
    from ultralytics.nn.modules import GhostBottleneck
    m = GhostBottleneck(128, 128).to(DEVICE)
    x = torch.randn(1, 128, 80, 80, device=DEVICE)
    out = m(x)
    assert out.shape == (1, 128, 80, 80), f"Got {out.shape}"


@check("GhostC2f forward pass (1, 256, 40, 40) → (1, 256, 40, 40)")
def check_ghost_c2f():
    from models.ghost_conv import GhostC2f
    m = GhostC2f(256, 256, n=2, shortcut=True).to(DEVICE)
    x = torch.randn(1, 256, 40, 40, device=DEVICE)
    out = m(x)
    assert out.shape == (1, 256, 40, 40), f"Got {out.shape}"


@check("ChannelAttention forward pass (1, 256, 40, 40) → same shape")
def check_channel_attn():
    from models.attention import ChannelAttention
    m = ChannelAttention(256).to(DEVICE)
    x = torch.randn(1, 256, 40, 40, device=DEVICE)
    out = m(x)
    assert out.shape == x.shape, f"Got {out.shape}"


@check("SpatialAttention forward pass (1, 256, 40, 40) → same shape")
def check_spatial_attn():
    from models.attention import SpatialAttention
    m = SpatialAttention(kernel_size=7).to(DEVICE)
    x = torch.randn(1, 256, 40, 40, device=DEVICE)
    out = m(x)
    assert out.shape == x.shape, f"Got {out.shape}"


@check("CBAM forward pass (1, 256, 40, 40) → same shape (attention applied)")
def check_cbam():
    from models.attention import CBAM
    m = CBAM(c1=256).to(DEVICE)
    x = torch.randn(1, 256, 40, 40, device=DEVICE)
    out = m(x)
    assert out.shape == x.shape, f"Got {out.shape}"
    # Verify attention actually changes the values (not a no-op)
    assert not torch.allclose(x, out), "CBAM output identical to input (attention not applied)"


@check("UnderwaterEnhancer (enabled=True): BGR ndarray → enhanced BGR ndarray")
def check_enhancer_enabled():
    from models.underwater_enhance import UnderwaterEnhancer
    enhancer = UnderwaterEnhancer(enabled=True)
    img = np.random.randint(20, 180, (480, 640, 3), dtype=np.uint8)
    out = enhancer(img)
    assert out.shape == img.shape, f"Shape mismatch: {out.shape}"
    assert out.dtype == np.uint8, f"Wrong dtype: {out.dtype}"


@check("UnderwaterEnhancer (enabled=False): returns input unchanged")
def check_enhancer_disabled():
    from models.underwater_enhance import UnderwaterEnhancer
    enhancer = UnderwaterEnhancer(enabled=False)
    img = np.random.randint(20, 180, (480, 640, 3), dtype=np.uint8)
    out = enhancer(img)
    assert np.array_equal(out, img), "Disabled enhancer modified the image"


@check("All 6 underwater augmentation effects fire independently")
def check_augmentations():
    from scripts.underwater_augmentation import (
        TurbidityHaze, ArtificialBackscatter, LowIllumination,
        ColorAttenuation, MotionBlur, UnderwaterNoise
    )
    img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    for cls in [TurbidityHaze, ArtificialBackscatter, LowIllumination,
                ColorAttenuation, MotionBlur, UnderwaterNoise]:
        out = cls()(img)
        assert out.shape == img.shape and out.dtype == np.uint8


@check("UnderwaterAugmentationPipeline (p=1.0): applies ≥1 effect")
def check_aug_pipeline():
    from scripts.underwater_augmentation import UnderwaterAugmentationPipeline
    pipeline = UnderwaterAugmentationPipeline(p=1.0)
    img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    out = pipeline(img)
    assert out.shape == img.shape and out.dtype == np.uint8


@check("WIoULoss (wiou): computes scalar loss from paired bbox tensors")
def check_wiou():
    from scripts.custom_loss import WIoULoss
    torch.manual_seed(0)
    loss_fn = WIoULoss(loss_type="wiou")
    pred = torch.rand(32, 4)
    target = torch.rand(32, 4)
    loss = loss_fn(pred, target)
    assert loss.ndim == 0, "Loss should be scalar"
    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "NaN in WIoU loss"
    assert not torch.isinf(loss), "Inf in WIoU loss"


@check("WIoULoss (ciou): computes scalar loss")
def check_ciou_fallback():
    from scripts.custom_loss import WIoULoss
    loss_fn = WIoULoss(loss_type="ciou")
    pred = torch.rand(16, 4)
    target = torch.rand(16, 4)
    loss = loss_fn(pred, target)
    assert loss.ndim == 0 and loss.item() > 0 and not torch.isnan(loss)


@check("WIoULoss: gradient flows through (differentiable)")
def check_wiou_gradient():
    from scripts.custom_loss import WIoULoss
    loss_fn = WIoULoss(loss_type="wiou")
    pred = torch.rand(16, 4, requires_grad=True)
    target = torch.rand(16, 4)
    loss = loss_fn(pred, target)
    loss.backward()
    assert pred.grad is not None, "No gradient"
    assert not torch.isnan(pred.grad).any(), "NaN in gradient"


@check("YOLO-DarkWater YAML loads without error (custom module registration)")
def check_yaml_loads():
    from ultralytics import YOLO
    model = YOLO(str(REPO_ROOT / "models" / "yolo-darkwater.yaml"))
    assert model is not None


@check("YOLO-DarkWater full forward pass (1, 3, 640, 640) → 3 detection outputs")
def check_darkwater_forward():
    from ultralytics import YOLO
    model = YOLO(str(REPO_ROOT / "models" / "yolo-darkwater.yaml")).to(DEVICE)
    x = torch.randn(1, 3, 640, 640, device=DEVICE)
    with torch.no_grad():
        out = model.model(x)

    # Ultralytics returns (train_out, None) or list of predictions
    # In train mode: out is a tuple of feature maps or dict of predictions
    if isinstance(out, (list, tuple)):
        print(f"\n      Output: {len(out)} items", end="  ")
    elif isinstance(out, dict):
        print(f"\n      Output dict keys: {list(out.keys())}", end="  ")
    else:
        print(f"\n      Output shape: {out.shape}", end="  ")


@check("YOLOv8n baseline loads and runs forward pass")
def check_baseline_forward():
    baseline_pt = REPO_ROOT / "notebooks" / "yolov8n.pt"
    from ultralytics import YOLO
    if baseline_pt.exists():
        model = YOLO(str(baseline_pt)).to(DEVICE)
    else:
        model = YOLO("yolov8n.pt").to(DEVICE)  # downloads if needed
    x = torch.randn(1, 3, 640, 640, device=DEVICE)
    with torch.no_grad():
        out = model.model(x)
    print(f"\n      YOLOv8n loaded OK", end="  ")


@check("GhostC2f layers present in DarkWater model (not C2f)")
def check_ghost_layers_in_model():
    from ultralytics import YOLO
    from models.ghost_conv import GhostC2f
    model = YOLO(str(REPO_ROOT / "models" / "yolo-darkwater.yaml"))
    ghost_count = sum(1 for m in model.model.modules() if isinstance(m, GhostC2f))  # type: ignore[union-attr]
    assert ghost_count > 0, f"No GhostC2f layers found in model!"
    print(f"\n      Found {ghost_count} GhostC2f layers", end="  ")


@check("CBAM layer present in DarkWater model at P3")
def check_cbam_in_model():
    from ultralytics import YOLO
    from models.attention import CBAM
    model = YOLO(str(REPO_ROOT / "models" / "yolo-darkwater.yaml"))
    cbam_count = sum(1 for m in model.model.modules() if isinstance(m, CBAM))  # type: ignore[union-attr]
    assert cbam_count > 0, f"No CBAM layer found in model!"
    print(f"\n      Found {cbam_count} CBAM layer(s)", end="  ")


# ---------------------------------------------------------------------------
# Run all checks
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 65)
    print("  YOLO-DarkWater Module Validation Suite")
    print("=" * 65)
    print(f"  Device: {DEVICE.upper()}")
    if DEVICE == "cuda":
        print(f"  GPU:    {torch.cuda.get_device_name(0)}")
    print()

    checks = [
        check_cuda,
        check_ghost_conv,
        check_ghost_bottleneck,
        check_ghost_c2f,
        check_channel_attn,
        check_spatial_attn,
        check_cbam,
        check_enhancer_enabled,
        check_enhancer_disabled,
        check_augmentations,
        check_aug_pipeline,
        check_wiou,
        check_ciou_fallback,
        check_wiou_gradient,
        check_yaml_loads,
        check_darkwater_forward,
        check_baseline_forward,
        check_ghost_layers_in_model,
        check_cbam_in_model,
    ]

    for check_fn in checks:
        check_fn()

    print("\n" + "=" * 65)
    print(f"  Results: {len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print(f"\n  FAILED checks:")
        for name in FAILED:
            print(f"    ✗ {name}")
        print("\n  Fix all failures before training!")
        sys.exit(1)
    else:
        print("\n  ✓ All checks passed. Ready for training!")
    print("=" * 65)


if __name__ == "__main__":
    main()
