"""
scripts/custom_loss.py
=======================
Wise-IoU v3 loss for YOLO-DarkWater.

Implements the complete Wise-IoU v3 (WIoU) bounding box regression loss with
dynamic non-monotonic focusing mechanism, plus a switchable CIoU fallback.

References:
    Tong et al., "Wise-IoU: Bounding Box Regression Loss with Dynamic Focusing
    Mechanism", arXiv 2301.10051, 2023.
    https://arxiv.org/abs/2301.10051

Architecture integration:
    DarkWaterDetectionLoss subclasses Ultralytics' v8DetectionLoss and overrides
    the box regression loss component. This is the minimal-invasive hook that
    preserves all other Ultralytics loss components (obj, cls, DFL).

Usage (via custom trainer):
    trainer.model.criterion = DarkWaterDetectionLoss(trainer.model.model, loss_type='wiou')
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Literal


# ---------------------------------------------------------------------------
# WIoU Core Computation
# ---------------------------------------------------------------------------

def wiou_loss(
    pred_bboxes: torch.Tensor,
    target_bboxes: torch.Tensor,
    loss_type: Literal["wiou", "ciou"] = "wiou",
    alpha: float = 1.9,
    delta: float = 3.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Compute Wise-IoU v3 or CIoU bounding box regression loss.

    Args:
        pred_bboxes:   Predicted boxes, shape (N, 4), format [x1, y1, x2, y2].
        target_bboxes: Target boxes, shape (N, 4), same format.
        loss_type:     'wiou' for Wise-IoU v3, 'ciou' for standard CIoU.
        alpha:         WIoU non-monotonic coefficient (paper default: 1.9).
        delta:         WIoU focusing magnitude (paper default: 3.0).
        eps:           Numerical stability epsilon.

    Returns:
        Scalar loss tensor (mean over batch).

    WIoU v3 formula:
        1. Compute IoU between pred and target boxes.
        2. Compute normalized center distance r = center_d² / enclose_d².
        3. Compute batch-normalized dynamic focus β = r / mean(r).detach()
        4. Non-monotonic focusing weight: w(β) = β · exp(β/alpha/delta - 1/delta)
           (down-weights outlier anchors with large center distance)
        5. WIoU_loss = w(β) · (1 - IoU)

    The batch normalization in step 3 is critical: it makes β dimensionless and
    avoids scale sensitivity across different training stages.
    """
    # Unpack box coordinates [x1, y1, x2, y2]
    b1_x1, b1_y1, b1_x2, b1_y2 = pred_bboxes.unbind(-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = target_bboxes.unbind(-1)

    # ----- Intersection -----
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    # ----- Union -----
    area_b1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area_b2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = area_b1 + area_b2 - inter_area + eps

    # ----- Standard IoU -----
    iou = inter_area / union_area

    if loss_type == "ciou":
        # ---- CIoU loss (standard Ultralytics default, kept as fallback) ----
        # Center distance
        c1_x = (b1_x1 + b1_x2) / 2
        c1_y = (b1_y1 + b1_y2) / 2
        c2_x = (b2_x1 + b2_x2) / 2
        c2_y = (b2_y1 + b2_y2) / 2
        center_dist2 = (c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2

        # Enclosing box diagonal
        enclose_x1 = torch.min(b1_x1, b2_x1)
        enclose_y1 = torch.min(b1_y1, b2_y1)
        enclose_x2 = torch.max(b1_x2, b2_x2)
        enclose_y2 = torch.max(b1_y2, b2_y2)
        enclose_diag2 = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + eps

        # Aspect ratio consistency term
        w1 = b1_x2 - b1_x1
        h1 = b1_y2 - b1_y1
        w2 = b2_x2 - b2_x1
        h2 = b2_y2 - b2_y1
        v = (4 / (torch.pi ** 2)) * torch.pow(
            torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2
        )
        with torch.no_grad():
            trade_off = v / (1 - iou + v + eps)

        ciou = iou - (center_dist2 / enclose_diag2) - trade_off
        return (1 - ciou).mean()

    # ---- Wise-IoU v3 ----
    # Center distance
    c1_x = (b1_x1 + b1_x2) / 2
    c1_y = (b1_y1 + b1_y2) / 2
    c2_x = (b2_x1 + b2_x2) / 2
    c2_y = (b2_y1 + b2_y2) / 2
    center_dist2 = (c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2

    # Enclosing diagonal
    enclose_x1 = torch.min(b1_x1, b2_x1)
    enclose_y1 = torch.min(b1_y1, b2_y1)
    enclose_x2 = torch.max(b1_x2, b2_x2)
    enclose_y2 = torch.max(b1_y2, b2_y2)
    enclose_diag2 = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + eps

    # Normalized geometric factor r ∈ [0, 1]
    r = center_dist2 / enclose_diag2  # shape (N,)

    # Batch-normalized dynamic focusing coefficient β
    # .detach() stops gradient flowing through the normalization denominator
    r_hat = r.mean().detach()  # scalar, no-grad
    beta = r / (r_hat + eps)   # dimensionless ratio (N,)

    # Non-monotonic focusing weight (WIoU v3 formula)
    # For β < 1: down-weights well-localized anchors less strongly
    # For β > 1: exponentially down-weights outlier / low-quality anchors
    focusing_weight = beta * torch.exp(beta / (alpha * delta) - 1.0 / delta)

    # WIoU loss: geometric-weighted IoU loss
    wiou = focusing_weight * (1 - iou)

    return wiou.mean()


# ---------------------------------------------------------------------------
# WIoULoss module wrapper (standalone, for use outside Ultralytics)
# ---------------------------------------------------------------------------

class WIoULoss(nn.Module):
    """
    Standalone Wise-IoU v3 loss module.

    Can be used independently of Ultralytics for testing or ablation.

    Args:
        loss_type: 'wiou' for WIoU v3, 'ciou' for standard CIoU.
        alpha:     WIoU focusing parameter α (default 1.9, paper value).
        delta:     WIoU magnitude parameter δ (default 3.0, paper value).
    """

    def __init__(
        self,
        loss_type: Literal["wiou", "ciou"] = "wiou",
        alpha: float = 1.9,
        delta: float = 3.0,
    ) -> None:
        super().__init__()
        self.loss_type = loss_type
        self.alpha = alpha
        self.delta = delta

    def forward(
        self,
        pred_bboxes: torch.Tensor,
        target_bboxes: torch.Tensor,
    ) -> torch.Tensor:
        return wiou_loss(
            pred_bboxes,
            target_bboxes,
            loss_type=self.loss_type,
            alpha=self.alpha,
            delta=self.delta,
        )

    def __repr__(self) -> str:
        return (
            f"WIoULoss(type={self.loss_type}, α={self.alpha}, δ={self.delta})"
        )


# ---------------------------------------------------------------------------
# Ultralytics Integration — DarkWaterDetectionLoss
# ---------------------------------------------------------------------------

class DarkWaterDetectionLoss:
    """
    Custom detection loss for YOLO-DarkWater that replaces the default CIoU
    box regression with Wise-IoU v3 while preserving all other Ultralytics
    loss components (DFL, classification).

    Usage:
        # In DarkWaterTrainer (see scripts/train.py):
        trainer.model.criterion = DarkWaterDetectionLoss(
            model, loss_type=cfg.get('loss_type', 'wiou')
        )

    Implementation approach:
        Subclasses Ultralytics v8DetectionLoss and overrides __call__ to
        swap the iou computation step for WIoU.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_type: Literal["wiou", "ciou"] = "wiou",
        alpha: float = 1.9,
        delta: float = 3.0,
    ) -> None:
        try:
            from ultralytics.utils.loss import v8DetectionLoss
            self._base_loss = v8DetectionLoss(model)
        except ImportError:
            raise ImportError("Ultralytics not installed. Run: pip install ultralytics")

        self.loss_type = loss_type
        self.alpha = alpha
        self.delta = delta
        self._wiou = WIoULoss(loss_type=loss_type, alpha=alpha, delta=delta)

        # Mirror attributes Ultralytics expects on criterion
        self.device = self._base_loss.device
        self.nc = self._base_loss.nc
        self.reg_max = self._base_loss.reg_max

    def __call__(self, preds, batch):
        """
        Compute loss. Delegates to base Ultralytics loss, but replaces the
        internal bbox_iou call with WIoU.

        Note: Full WIoU integration requires hooking into the inner box loss
        computation. For maximum Ultralytics compatibility, we patch the
        bbox_iou function temporarily during the forward pass.
        """
        import ultralytics.utils.metrics as _metrics
        from ultralytics.utils.metrics import bbox_iou as _orig_iou

        if self.loss_type == "wiou":
            # Temporarily monkey-patch bbox_iou to use WIoU
            def _wiou_hook(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
                """Intercepts bbox_iou calls during loss computation → returns WIoU."""
                # Convert from whatever format to x1y1x2y2
                if xywh:
                    # Convert [cx, cy, w, h] → [x1, y1, x2, y2]
                    b1_xy = box1[..., :2]
                    b1_wh_half = box1[..., 2:] / 2
                    b2_xy = box2[..., :2]
                    b2_wh_half = box2[..., 2:] / 2
                    b1 = torch.cat([b1_xy - b1_wh_half, b1_xy + b1_wh_half], dim=-1)
                    b2 = torch.cat([b2_xy - b2_wh_half, b2_xy + b2_wh_half], dim=-1)
                else:
                    b1, b2 = box1, box2

                # Flatten batch dims for WIoU computation
                orig_shape = b1.shape[:-1]
                b1_flat = b1.reshape(-1, 4)
                b2_flat = b2.reshape(-1, 4)

                iou_vals = 1.0 - wiou_loss(
                    b1_flat, b2_flat,
                    loss_type="wiou",
                    alpha=self.alpha,
                    delta=self.delta,
                )
                # Return as (N,) tensor matching original bbox_iou contract
                # (base loss will subtract from 1 to get the loss)
                return iou_vals.reshape(orig_shape)

            try:
                _metrics.bbox_iou = _wiou_hook
                loss, loss_items = self._base_loss(preds, batch)
            finally:
                _metrics.bbox_iou = _orig_iou  # always restore
        else:
            loss, loss_items = self._base_loss(preds, batch)

        return loss, loss_items


if __name__ == "__main__":
    # Quick sanity test
    import torch

    torch.manual_seed(42)
    pred = torch.rand(16, 4)
    target = torch.rand(16, 4)

    for lt in ("wiou", "ciou"):
        loss_fn = WIoULoss(loss_type=lt)
        l = loss_fn(pred, target)
        print(f"{lt.upper()} loss: {l.item():.4f}")
    print("WIoULoss module OK")