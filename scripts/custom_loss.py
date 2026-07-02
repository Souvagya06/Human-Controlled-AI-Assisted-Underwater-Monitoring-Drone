import torch
import torch.nn as nn

class WIoULoss(nn.Module):
    def __init__(self):
        super(WIoULoss, self).__init__()

    def forward(self, pred_bboxes, target_bboxes):
        """
        Calculates the Wise-IoU (WIoU) loss.
        Expects inputs in [x1, y1, x2, y2] format.
        """
        b1_x1, b1_y1, b1_x2, b1_y2 = pred_bboxes.chunk(4, dim=-1)
        b2_x1, b2_y1, b2_x2, b2_y2 = target_bboxes.chunk(4, dim=-1)

        # 1. Standard IoU Calculation
        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        area_b1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area_b2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union_area = area_b1 + area_b2 - inter_area + 1e-7
        
        iou = inter_area / union_area

        # 2. Distance Penalty (Center distance over enclose diagonal)
        c1_x, c1_y = (b1_x1 + b1_x2) / 2, (b1_y1 + b1_y2) / 2
        c2_x, c2_y = (b2_x1 + b2_x2) / 2, (b2_y1 + b2_y2) / 2
        center_distance = (c1_x - c2_x)**2 + (c1_y - c2_y)**2
        
        enclose_x1 = torch.min(b1_x1, b2_x1)
        enclose_y1 = torch.min(b1_y1, b2_y1)
        enclose_x2 = torch.max(b1_x2, b2_x2)
        enclose_y2 = torch.max(b1_y2, b2_y2)
        enclose_diagonal = (enclose_x2 - enclose_x1)**2 + (enclose_y2 - enclose_y1)**2 + 1e-7

        # 3. Dynamic Non-Monotonic Focusing Mechanism
        R_distance = center_distance / enclose_diagonal
        beta = R_distance / (iou + 1e-7)
        alpha, delta = 1.9, 3.0
        
        non_monotonic_focus = beta / (delta * torch.exp(beta / alpha))
        
        # Final WIoU Loss computation
        wiou_loss = (1 - iou) * torch.exp(R_distance) * non_monotonic_focus
        return wiou_loss.mean()