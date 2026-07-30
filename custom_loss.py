import torch
import torch.nn as nn
import math

class WiseIoULoss(nn.Module):
    def __init__(self, alpha=1.9, delta=3.0, reduction='mean'):
        super(WiseIoULoss, self).__init__()
        self.alpha = alpha
        self.delta = delta
        self.reduction = reduction

    def forward(self, pred, target, bbox_weight=None):
        # pred, target: [N, 4] (x1, y1, x2, y2) or cxcywh depending on integration
        # Standard WiSIM / Wiou calculation implementation for bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

        # Intersection area
        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)
        inter = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

        # Union area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
        union = w1 * h1 + w2 * h2 - inter + 1e-7

        iou = inter / union

        # Enclosing box
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        c2 = cw ** 2 + ch ** 2 + 1e-7

        # Center distance
        bx1, by1 = (b1_x1 + b1_x2) / 2, (b1_y1 + b1_y2) / 2
        bx2, by2 = (b2_x1 + b2_x2) / 2, (b2_y1 + b2_y2) / 2
        rho2 = (bx2 - bx1) ** 2 + (by2 - by1) ** 2

        # Wise-IoU v3 components
        r_wiou = torch.exp(rho2 / c2)
        
        # Non-monotonic focus coefficient beta
        # beta = \frac{\rho^2}{c^2}^\alpha / \delta ... simplified formulation
        with torch.no_grad():
            beta = torch.pow(rho2 / c2, self.alpha) / self.delta
            beta = torch.clamp(beta, min=1e-7)

        wiou = iou * torch.exp((rho2 / c2) / self.delta) # approximation for stable gradient scaling
        loss = beta * (1 - iou) * torch.exp(rho2 / c2)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss