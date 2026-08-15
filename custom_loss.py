import torch
import torch.nn.functional as F
from torch import nn
from ultralytics.utils.loss import v8DetectionLoss, BboxLoss
from ultralytics.utils.tal import bbox2dist


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

        r = torch.pow(rho2 / c2, self.alpha) / self.delta
        r = torch.clamp(r, min=1e-7)
        r_hat = r.mean().detach()
        beta = torch.pow(r / r_hat, self.alpha - 1.0)

        loss = beta * (1.0 - iou) * torch.exp(rho2 / c2)
        return loss, r_hat


class WiseIoUBboxLoss(BboxLoss):
    def __init__(self, reg_max=16, alpha=1.9, delta=3.0):
        super().__init__(reg_max)
        self.wiou_loss = WiseIoULoss(alpha=alpha, delta=delta)

    def forward(
        self,
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
        imgsz,
        stride,
    ):
        # Compute Wise-IoU over ALL anchors so r_hat is batch-normalized across the full anchor set
        loss_iou_all, _ = self.wiou_loss(pred_bboxes, target_bboxes)

        weight = target_scores[fg_mask].sum(-1, keepdim=True)
        loss_iou = (loss_iou_all[fg_mask] * weight).sum() / target_scores_sum

        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl


class CustomDetectionLoss(v8DetectionLoss):
    def __init__(self, model, tal_topk=10, tal_topk2=None, wiou_alpha=1.9, wiou_delta=3.0):
        super().__init__(model, tal_topk, tal_topk2)
        self.bbox_loss = WiseIoUBboxLoss(self.reg_max, alpha=wiou_alpha, delta=wiou_delta).to(self.device)
