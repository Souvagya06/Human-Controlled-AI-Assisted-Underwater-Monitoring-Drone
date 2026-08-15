# Human-Controlled AI-Assisted Underwater Monitoring Drone: YOLO Custom Model

An advanced object detection system built on YOLOv8, purpose-built for underwater environments with severe visibility degradation, color attenuation, backscatter, and murky lighting. The system combines a rigorous preprocessing pipeline, automated data ingestion, custom architectural modules, and a standardized training workflow.

---

## 1. Project Workflow

The project follows a notebook-driven pipeline from raw COCO data to trained custom model, with validation at each stage:

```text
01_underwater_preprocessing.ipynb  -->  Refined Moody Spotlight Pipeline + enhanced dataset generation
02_train_baseline.ipynb             -->  Standard YOLOv8n baseline training with reproducible seed/patience
03_train_custom_v1.ipynb            -->  Custom YAML architecture with CBAM + BiFPN + Wise-IoU v3 + pre-training checklist
smoke_test.py                       -->  Pre-training unit tests (CBAM shape, Wise-IoU v3 gradient flow, YAML load)
prepare_data.py                     -->  COCO-to-YOLO conversion, coordinate normalization, dataset.yaml generation
audit_dataset.py                    -->  Dataset label audit (class distribution, empty files)
```

---

## 2. Folder Structure

```text
underwater_drone_custom_model/
├── dataset/
│   ├── annotations/                 # Original COCO JSON files
│   ├── images/{train,val}/
│   └── labels/{train,val}           # Converted YOLO .txt labels
├── dataset_enhanced/
│   ├── images/{train,val}           # Refined Moody Spotlight Pipeline output
│   └── labels/{train,val}           # Copied YOLO .txt labels
├── original_data/
│   ├── annotations/
│   └── images/
├── notebooks/
│   ├── 01_underwater_preprocessing.ipynb
│   ├── 02_train_baseline.ipynb
│   └── 03_train_custom_v1.ipynb
├── attention.py                     # CBAM + ChannelAttention + SpatialAttention
├── custom_loss.py                   # WiseIoULoss, WiseIoUBboxLoss, CustomDetectionLoss
├── ghost_conv.py                    # GhostConv / GhostC2f
├── models_init.py                   # Custom module registration into ultralytics.nn.tasks
├── yolo-custom.yaml                 # Custom architecture: P2 head + BiFPN + CBAM
├── dataset.yaml                     # Raw dataset config
├── dataset_enhanced.yaml            # Enhanced dataset config
├── prepare_data.py                  # Automated COCO → YOLO ingestion
├── smoke_test.py                    # CBAM + WiseIoU + YAML pre-training validation
├── audit_dataset.py                 # Label audit script
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 3. Data Preparation (`prepare_data.py`)

Automated ingestion pipeline that converts raw COCO annotations into YOLOv8-ready format:

- **Dataset discovery** — locates COCO JSON files and raw image directories automatically
- **COCO → YOLO conversion** — parses `[x_min, y_min, w, h]` to normalized `[class_id, x_center, y_center, width, height]`
- **Coordinate bounds checking** — clamps normalized coordinates to `[0.0, 1.0]`
- **Directory structuring** — creates standard YOLOv8 `images/{train,val}` and `labels/{train,val}` hierarchy
- **Auto-generated `dataset.yaml`** — extracts class names from COCO metadata and writes path mappings + class count

---

## 4. Preprocessing Pipeline: Refined Moody Spotlight

Applied in `01_underwater_preprocessing.ipynb` before training:

1. **Grayworld Color Balance** — uniform B/G/R scaling to remove blue-green cast
2. **LAB CLAHE** — contrast-limited adaptive histogram equalization on the L-channel
3. **Gamma Correction (γ = 0.55)** — LUT-based midtone suppression for deep moody spotlight effect
4. **Red Channel Boost (+8)** — restores biological warmth lost to wavelength attenuation
5. **Advanced Denoising** — `fastNlMeansDenoisingColored` + 3×3 median blur
6. **Unsharp Masking** — Gaussian blur + `cv2.addWeighted` for sharp target edges

---

## 5. Custom Source Modules

### `attention.py` — CBAM Attention Module
Implements Convolutional Block Attention Module (CBAM):
- **ChannelAttention** — adaptive avg/max pooling → shared MLP bottleneck → sigmoid activation
- **SpatialAttention** — channel-wise avg/max pooling → convolution → sigmoid activation
- **CBAM** — sequential channel + spatial re-weighting block
- **Lazy MLP initialization** — channel dimension is inferred at runtime from input shape, allowing the block to survive automatic width scaling

### `custom_loss.py` — Wise-IoU v3 Loss
Custom bounding-box regression loss for YOLOv8 training:
- **WiseIoULoss** — per-anchor Wise-IoU v3 computation with batch-normalized reference value `r_hat = r.mean().detach()`
- **WiseIoUBboxLoss** — replaces default CIoU in the bbox loss head while preserving DFL logic
- **CustomDetectionLoss** — integrates WiseIoUBboxLoss into the Ultralytics training loop by subclassing `v8DetectionLoss`

### `ghost_conv.py` — Ghost Convolution
Efficient feature-map generation using GhostConv and GhostC2f blocks to reduce computational cost while preserving representational capacity.

### `models_init.py` — Module Registration
Registers custom modules (`GhostC2f`, `CBAM`) into `ultralytics.nn.tasks` so the YAML parser can resolve them by string name during model building.

---

## 6. Custom Architecture (`yolo-custom.yaml`)

Extends YOLOv8n with the following design:

- **Backbone** — standard Conv + C2f stages, P1/2 → P5/32
- **BiFPN-style neck** — top-down weighted feature fusion + bottom-up lateral links
- **CBAM** — injected at P3/8 scale to refine medium-resolution features before the P2 upsampling branch
- **P2 head** — 4× finer resolution (`P2/4`) for tiny debris and small fish detection
- **Detection head** — 4-scale `Detect` on P2, P3, P4, P5
- **Scales** — `n: [0.33, 0.25, 1024]` (depth, width, max_channels)
- **Classes** — `nc: 16`

---

## 7. Pre-Training Checklist

### Baseline (`02_train_baseline.ipynb`)

1. **Data sanity** — confirms `dataset.yaml` exists and `images/train`, `images/val` directories are present; halts with clear error if missing
2. **Reproducibility** — `seed=42`, `patience=35` for apples-to-apples comparison with the custom run

### Custom (`03_train_custom_v1.ipynb`)

1. **Data sanity** — confirms `dataset_enhanced.yaml` + `images/train` + `images/val`; halts on failure
2. **Module registration** — `models_init.register_custom_modules()` before model build
3. **Loss hook** — `tasks.v8DetectionLoss = custom_loss.CustomDetectionLoss` before training
4. **Model build** — `YOLO('../yolo-custom.yaml').load('yolov8n.pt')`
5. **Forward-pass sanity** — dummy `torch.randn(1, 3, 640, 640)` through the model; prints output shapes
6. **Custom layer verification** — counts `CBAM` instances in the built model and asserts presence
7. **Gradient smoke test** — runs a fake batch through `model.model.criterion(preds, fake_batch)`; calls `loss.sum().backward()`; asserts no NaN; prints loss value and per-component breakdown
8. **Hyperparameter correctness** — `multi_scale=0.5` (float scale factor); `copy_paste` omitted (bbox-only dataset); `seed=42`, `patience=35`
9. **Inference guidance** — documents correct `model.predict(source=...)` / `model(img)` usage and warns against calling the inner tensor-only `model.model.model` with file paths

---

## 8. Smoke Tests (`smoke_test.py`)

Three unit tests run before any training:

1. **CBAM shape test** — verifies CBAM preserves spatial dimensions `(2, 256, 32, 32) → (2, 256, 32, 32)`
2. **Wise-IoU v3 gradient flow** — computes loss, runs `loss.mean().backward()`, asserts no NaN, prints `r_hat`
3. **YAML model load** — confirms the custom architecture builds successfully through the Ultralytics `YOLO()` parser

---

## 9. Training Configuration

### Baseline (`02_train_baseline.ipynb`)
- **Model** — `yolov8n.pt`
- **Data** — `dataset.yaml`
- **Epochs** — 110
- **Seed / Patience** — 42 / 35
- **Output** — `runs/detect/underwater_refined_yolov8n/weights/best.pt`

### Custom (`03_train_custom_v1.ipynb`)
- **Model** — `yolo-custom.yaml` + `yolov8n.pt` weights
- **Data** — `dataset_enhanced.yaml`
- **Epochs** — 150
- **Seed / Patience** — 42 / 35
- **Optimizer** — AdamW, `lr0=0.0025`, cosine LR
- **AMP / EMA** — enabled
- **Multi-scale** — `multi_scale=0.5` (samples image sizes from 320 to 960 px)
- **Augmentations** — `mosaic=1.0`, `fliplr=0.5`, `mixup=0.1`, `hsv_h=0.0`, `hsv_s=0.02`, `hsv_v=0.02` (protects Grayworld color balance)
- **Output** — `runs/detect/underwater_custom_v1/weights/best.pt`

---

## 10. Methodology Note

Baseline trains on raw `dataset.yaml`; custom trains on enhanced `dataset_enhanced.yaml`. Enhancement is treated as part of the custom pipeline. To isolate architecture/loss effects from preprocessing effects in a future ablation, run the custom architecture on raw data or the baseline on enhanced data.

---

## 11. Requirements

```text
ultralytics>=8.0.0
torch>=2.0.0
opencv-python>=4.8.0
numpy>=1.24.0
pyyaml>=6.0
tqdm
```

---

## 12. Quick Start

```bash
pip install -r requirements.txt

python prepare_data.py
python smoke_test.py
jupyter notebook notebooks/02_train_baseline.ipynb
jupyter notebook notebooks/03_train_custom_v1.ipynb
```

---

## 13. Current Status

- [x] Project initialization and folder structure
- [x] Data preparation (`prepare_data.py`) and COCO-to-YOLO conversion
- [x] Dataset audit (`audit_dataset.py`)
- [x] Refined Moody Spotlight preprocessing pipeline
- [x] Baseline training notebook with data sanity + reproducible seed/patience
- [x] Custom YAML architecture (`yolo-custom.yaml`) with P2 + BiFPN + CBAM
- [x] CBAM attention module with lazy MLP init
- [x] Ghost convolution implementation
- [x] Module registration (`models_init.py`)
- [x] Wise-IoU v3 loss with batch-normalized `r_hat`
- [x] Loss hook into Ultralytics training loop (`CustomDetectionLoss`)
- [x] Smoke test suite (CBAM shape, WiseIoU gradient, YAML load)
- [x] 9-point pre-training checklist in custom notebook
- [x] Gradient smoke test + custom layer count verification
- [x] Inference usage guidance in custom notebook

---

*Last Updated: 2026-08-15*
