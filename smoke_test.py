import torch
from ultralytics import YOLO
from attention import CBAM
from custom_loss import WiseIoULoss
import ultralytics.nn.tasks as tasks
tasks.__dict__['CBAM'] = CBAM

def run_smoke_test():
    print("Running pre-training smoke tests...")
    
    cbam = CBAM(c1=256)
    x = torch.randn(2, 256, 32, 32)
    out = cbam(x)
    assert out.shape == x.shape, f"CBAM shape mismatch: got {out.shape}, expected {x.shape}"
    print("[PASS] CBAM attention shape test passed.")

    wiou = WiseIoULoss()
    preds = torch.tensor([[10.0, 10.0, 50.0, 50.0], [20.0, 20.0, 60.0, 60.0]], requires_grad=True)
    targets = torch.tensor([[12.0, 12.0, 48.0, 48.0], [18.0, 18.0, 58.0, 58.0]])
    
    loss = wiou(preds, targets)
    loss.backward()
    assert not torch.isnan(loss), "Wise-IoU loss returned NaN!"
    print("[PASS] Wise-IoU v3 gradient flow and NaN check passed.")

    model = YOLO('yolo-custom.yaml')
    print("[PASS] Custom YOLO architecture YAML loaded successfully.")

if __name__ == "__main__":
    run_smoke_test()