import torch
import torch.nn as nn
from ultralytics.utils.loss import v8DetectionLoss

class WiseIoULoss(nn.Module):
    def __init__(self, alpha=1.9, delta=3.0):
        super().__init__()
        self.alpha = alpha
        self.delta = delta

    def forward(self, pred, target):
        b1_x1, b1_y1, b1_x2, b1_y2 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)
        inter = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
        union = w1 * h1 + w2 * h2 - inter + 1e-7
        iou = inter / union

        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        c2 = cw ** 2 + ch ** 2 + 1e-7

        bx1, by1 = (b1_x1 + b1_x2) / 2, (b1_y1 + b1_y2) / 2
        bx2, by2 = (b2_x1 + b2_x2) / 2, (b2_y1 + b2_y2) / 2
        rho2 = (bx2 - bx1) ** 2 + (by2 - by1) ** 2

        with torch.no_grad():
            r = torch.pow(rho2 / c2, self.alpha) / self.delta
            r = torch.clamp(r, min=1e-7)
            r_hat = r.mean().detach()
            beta = torch.pow(r / r_hat, self.alpha - 1.0)

        loss = beta * (1.0 - iou) * torch.exp(rho2 / c2)
        return loss.mean()

class CustomDetectionLoss(v8DetectionLoss):
    def __init__(self, model):
        super().__init__(model)
        self.wiou_loss = WiseIoULoss()

    def __call__(self, preds, batch):
        loss, loss_items = super().__call__(preds, batch)
        return loss, loss_items