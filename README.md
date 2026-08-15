# Human-Controlled AI-Assisted Underwater Monitoring Drone: YOLO Custom Model

The YOLO Custom Model is an advanced object detection iteration built on top of the YOLOv8 architecture, specifically designed to tackle the severe visibility degradation and color attenuation found in deep-sea environments.

Unlike standard models trained on generic imagery or simple contrast adjustments, this model integrates a meticulously crafted, multi-step underwater computer vision preprocessing pipeline combined with automated COCO-to-YOLO conversion and coordinate normalization. Furthermore, it incorporates state-of-the-art architectural upgrades (Tier 1–3 enhancements) including a **P2 small-object detection head**, **CBAM (Convolutional Block Attention Module)**, **BiFPN weighted feature fusion**, **Wise-IoU v3 loss**, and **Copy-Paste augmentation**. This ensures that the network learns to detect objects such as marine species and underwater debris accurately even when surrounded by heavy backscatter, chromatic casts, and murky lighting.

---

## 1. Project Workflow & Progress

The project follows a rigorous, notebook-driven development cycle:

```text
01_underwater_preprocessing.ipynb  -->  Refined Moody Spotlight Pipeline + enhanced dataset generation
02_train_baseline.ipynb             -->  Standard YOLOv8n baseline training on raw + enhanced data
03_train_custom_v1.ipynb            -->  Custom architecture training (P2 head + CBAM + BiFPN + Wise-IoU)
smoke_test.py                       -->  Pre-training unit tests for CBAM, WiseIoU, and YAML model loading
```

---

## 2. Final Project Folder Structure

The workspace is organized into a clean and structured layout to keep raw data, enhanced datasets, custom configurations, source modules, and notebooks completely isolated.

```text
underwater_drone_custom_model/
├── dataset/                       <-- Processed dataset (Train/Val + YOLO Labels)
│   ├── annotations/               <-- Original COCO JSON files
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
├── dataset_enhanced/              <-- Preprocessed dataset ready for YOLO training
│   ├── images/
│   │   ├── train/                 <-- Fully enhanced with Refined Moody Spotlight Pipeline
│   │   └── val/                   <-- Fully enhanced with Refined Moody Spotlight Pipeline
│   └── labels/                    <-- Copied YOLO label text files (.txt)
├── notebooks/                     <-- Jupyter Notebooks directory
│   ├── 01_underwater_preprocessing.ipynb
│   ├── 02_train_baseline.ipynb
│   └── 03_train_custom_v1.ipynb   <-- Advanced training script with Tier 1-3 injections
├── original_data/                 <-- Raw source dataset archive files
│   ├── annotations/
│   └── images/
├── attention.py                   <-- Custom CBAM and channel/spatial attention modules
├── custom_loss.py                 <-- Custom Wise-IoU v3 bounding box regression loss
├── ghost_conv.py                  <-- Ghost convolution implementation for efficient feature extraction
├── models_init.py                 <-- Custom model initialization and registration utilities
├── yolo-custom.yaml               <-- Custom YOLOv8 architecture configuration (P2 + BiFPN + CBAM)
├── dataset.yaml                   <-- Standard dataset configuration file
├── dataset_enhanced.yaml          <-- Enhanced dataset configuration mapping paths
├── prepare_data.py                <-- Automated data ingestion and coordinate script
├── smoke_test.py                  <-- Pre-training validation: CBAM shape, WiseIoU gradient, YAML load
├── requirements.txt               <-- Project dependency definitions
├── .gitignore                     <-- Git exclusion rules for large datasets/weights
└── README.md                      <-- Project documentation
```

The structured layout ensures that all raw data files, preprocessed outputs, architecture definitions, custom training scripts, and validation tests remain cleanly separated and easily maintainable across different development branches.

---

## 3. Data Preparation & Ingestion Methods (`prepare_data.py`)

Before image enhancement, raw data undergoes an automated processing pipeline to structure and convert annotations into standard YOLO format:

- **Dataset Discovery & Verification** — Automatically scanned and searched directory paths to locate COCO-format JSON annotation files and raw image directories.
- **COCO to YOLO Label Conversion** — Parsed JSON bounding boxes (`[x_min, y_min, w, h]`) and converted them into normalized YOLO format (`[class_id, x_center, y_center, width, height]`) relative to image dimensions.
- **Coordinate Bounds Checking** — Applied strict limiting functions (`min(max(val, 0.0), 1.0)`) to prevent out-of-bounds normalized bounding-box coordinates.
- **Directory Structuring** — Programmatically created and organized standard YOLOv8 folder hierarchies (`dataset/images/train`, `dataset/images/val`, `dataset/labels/train`, and `dataset/labels/val`).
- **Automated Dataset Configuration (`dataset.yaml`)** — Extracted category names dynamically from the JSON metadata and auto-generated the required YAML configuration file containing path mappings and class counts.

---

## 4. Preprocessing & Enhancement Methods Used in "YOLO Custom Model"

To optimize visual inputs for the YOLO Custom Model, images undergo the **Refined Moody Spotlight Pipeline**, which combines six distinct computer vision techniques implemented using OpenCV and NumPy:

- **Grayworld Color Balance** — Computes the mean intensities of the Blue, Green, and Red channels and scales them uniformly to eliminate the dominant blue-green chromatic cast typical of underwater imagery.
- **LAB Color Space CLAHE** — Converts the color-balanced frame into the LAB color space and applies Contrast Limited Adaptive Histogram Equalization (CLAHE) specifically to the luminance channel, boosting local target contrast without amplifying background specular noise.
- **Balanced Gamma Correction** — Applies a precise gamma transformation (γ = 0.55) through a lookup table (LUT) to suppress midtone haze and create a deep, moody spotlight effect where the background falls into shadow while targets stand out.
- **Controlled Red Channel Boost** — Mildly increases intensity values in the red channel (ΔR = +8) to restore natural biological color warmth lost due to underwater light wavelength attenuation.
- **Advanced Denoising** — Filters out high-frequency granular noise and floating particulate "snow" using Non-Local Means Denoising (`fastNlMeansDenoisingColored`) combined with a 3×3 median blur.
- **Crisp Unsharp Masking** — Combines a Gaussian blur filter with weighted image addition (`cv2.addWeighted`) to sharpen target boundaries and contours for optimal neural network feature extraction.

---

## 5. Custom Source Modules

### 5.1 `attention.py` — CBAM Attention Module
Implements **Convolutional Block Attention Module (CBAM)** with two sequential sub-modules:
- **ChannelAttention** — Uses adaptive average and max pooling followed by a shared MLP bottleneck (with configurable reduction ratio) and sigmoid activation to generate channel-wise attention maps.
- **SpatialAttention** — Uses channel-wise average and max pooling followed by a convolution to generate spatial attention maps.
- **CBAM** — Sequentially applies channel attention then spatial attention as a re-weighting block.

> **Note:** `ChannelAttention` uses lazy MLP initialization to support dynamic channel widths during Ultralytics automatic width scaling.

### 5.2 `custom_loss.py` — Wise-IoU v3 Loss
Implements **Wise-IoU v3** bounding box regression loss with:
- Dynamic bounding box format handling (xyxy / xywh)
- Distance-IoU (DIoU) and Complete-IoU (CIoU) components
- Wise-IoU v3 focusing mechanism to reduce gradient contribution from high-quality anchors
- Automatic NaN protection and numerical stability safeguards

### 5.3 `ghost_conv.py` — Ghost Convolution
Implements **GhostConv** to reduce computational cost by generating more feature maps from a smaller number of convolutions, using depthwise separable operations for efficient feature extraction.

### 5.4 `models_init.py` — Model Initialization Utilities
Provides utilities to register custom modules and initialize model components for seamless integration with the Ultralytics training pipeline.

---

## 6. Custom YAML Architecture (`yolo-custom.yaml`)

The custom model architecture extends YOLOv8n with the following modifications:

```yaml
# Backbone (P1 to P5)
Conv + C2f stages with standard YOLOv8n scaling

# Neck (BiFPN-style)
- Top-down pathway with weighted Concat and C2f blocks
- CBAM attention injected at P3/8 scale (256 channels base)
- Bottom-up pathway linking P2, P3, P4, P5
- Dedicated P2/4 small-object resolution branch (128 channels base)

# Detection Head
- 4-scale detection: P2/4, P3/8, P4/16, P5/32
- Scales: n=[0.33, 0.25, 1024] (depth, width, max_channels)
- nc: 16 classes
```

**Key architectural decisions:**
- CBAM is placed at the P3 scale to refine medium-resolution features before the P2 upsampling branch
- P2 head provides 4x finer resolution for tiny debris/fish detection
- BiFPN-style weighted fusion replaces plain concatenation in the neck

---

## 7. Smoke Test & Debugging (`smoke_test.py`)

The smoke test validates three critical components before training:

1. **CBAM Shape Test** — Verifies that CBAM preserves spatial dimensions (2, 256, 32, 32) → (2, 256, 32, 32)
2. **Wise-IoU Gradient Flow** — Verifies loss computes without NaN and gradients flow correctly through the network
3. **YAML Model Loading** — Verifies that the custom YAML architecture builds successfully with Ultralytics

### Issues Fixed During Smoke Test

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| `KeyError: 'CBAM'` | Ultralytics `parse_model()` resolves module names via `globals()['CBAM']`, but `CBAM` was not registered in `ultralytics.nn.tasks` namespace | Registered `CBAM` in `ultralytics.nn.tasks.__dict__` before `YOLO()` initialization |
| `IndexError: list index out of range` | Detection head `from` indices `[19, 23, 26, 29]` referenced non-existent layers after CBAM insertion | Corrected to `[19, 22, 25, 28]` to match actual layer numbering |
| `KeyError: 'nc'` / scale warning | YAML had `nc` and `scales` nested under `parameters:` instead of top-level keys | Moved `nc` and `scales` to top-level YAML keys |
| `TypeError: unsupported operand type(s) for +: 'NoneType' and 'int'` | Ultralytics width scaling was not propagating to custom modules, leaving `nc=None` | Fixed by ensuring top-level `nc` in YAML |
| `RuntimeError: expected input[...] to have 256 channels, but got 64 channels` | `ChannelAttention` MLP was statically built for 256 channels, but Ultralytics scales width to 64 at scale 'n' | Implemented lazy MLP initialization based on runtime input channel count |

---

## 8. Advanced Architecture & Training Injections (Tiers 1–3)

To maximize performance on tiny targets, severe class imbalances, and complex underwater backgrounds, the following enhancements have been injected directly into the custom pipeline:

- **Tier 1 (Core Foundations)** — CBAM attention injected at the P3 scale, Wise-IoU v3 bounding box regression, a dedicated P2 small-object detection head, Cosine Learning Rate scheduling, EMA, AMP, and disabled/restricted heavy HSV color augmentations.
- **Tier 2 (Advanced Feature Fusion & Loss)** — BiFPN weighted feature fusion in the neck to replace plain concatenation, and multi-scale training support for enhanced scale-invariance.
- **Tier 3 (Robust Augmentations)** — Copy-Paste augmentation to heavily diversify rare marine debris and class instances.

---

## 9. Model Comparison: Normal YOLOv8n Base Model vs. YOLO Custom Model

- **Input Data** — The Normal YOLOv8n Base Model works with raw, unprocessed underwater images that carry severe color casts, whereas the YOLO Custom Model works with professionally enhanced images produced through the Refined Moody Spotlight Pipeline.
- **Handling Backscatter** — The base model struggles with floating particles ("snow") and murky fog in the water, whereas the Custom Model actively filters out background snow using NLM denoising combined with deep gamma shadows.
- **Object Contrast** — The base model suffers from low local contrast, causing small or camouflaged objects to blend into the background. The Custom Model achieves high local contrast through LAB CLAHE and unsharp masking, making objects stand out sharply.
- **Detection Performance** — The base model shows higher false negative and false positive rates due to poor visibility and obscured features. The Custom Model delivers optimized bounding box localization, enhanced small-object detection via the P2 head, and a higher mean Average Precision (mAP) on underwater targets.

---

## 10. Model Comparison: "YOLO Dark Water" vs. "YOLO Custom Model"

- **YOLO Dark Water** — Focused entirely on robust data engineering, including automated dataset discovery, label parsing, bounding-box normalization, coordinate bounds checking, and directory structure generation. Its core methodology relied on structural preparation, coordinate integrity, and standardized YAML generation for building a reliable training baseline dataset.
- **YOLO Custom Model** — Extends everything achieved in the data ingestion framework and builds a sophisticated visual enhancement layer combined with advanced network architecture modifications on top of it. While YOLO Dark Water perfected structural preparation, the YOLO Custom Model introduces both the pixel-level Refined Moody Spotlight Pipeline and state-of-the-art Tier 1–3 architectural upgrades (BiFPN, CBAM, P2 head, and Wise-IoU) to systematically maximize detection accuracy in real-world underwater environments.

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

## 12. Usage

### Run Smoke Test
```bash
python smoke_test.py
```

### Prepare Data
```bash
python prepare_data.py
```

### Train Baseline
```bash
jupyter notebook notebooks/02_train_baseline.ipynb
```

### Train Custom Model
```bash
jupyter notebook notebooks/03_train_custom_v1.ipynb
```

---

## 13. Current Status

- [x] Project initialization and folder structure
- [x] Data preparation and COCO-to-YOLO conversion
- [x] Refined Moody Spotlight preprocessing pipeline
- [x] Baseline YOLOv8n training notebook
- [x] Custom architecture definition (`yolo-custom.yaml`)
- [x] CBAM attention module implementation
- [x] Wise-IoU v3 loss implementation
- [x] Ghost convolution implementation
- [x] Smoke test suite with CBAM, WiseIoU, and YAML validation
- [x] Debugged and resolved Ultralytics integration issues (module registration, YAML schema, lazy initialization)

---

*Last Updated: 2026-08-15*
