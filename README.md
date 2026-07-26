# 🌊 Human-Controlled AI-Assisted Underwater Monitoring Drone

## 1. Introduction to Our Second Model: "YOLO Custom Model"

The **YOLO Custom Model** is an advanced object detection iteration built on top of the **YOLOv8 architecture**, specifically designed to tackle the severe visibility degradation and color attenuation found in deep-sea environments.

Unlike standard models trained on generic imagery or simple contrast adjustments, this model integrates a meticulously crafted, multi-step underwater computer vision preprocessing pipeline combined with automated COCO-to-YOLO conversion and coordinate normalization. This ensures that the network learns to detect objects such as marine species and underwater debris accurately even when surrounded by heavy backscatter, chromatic casts, and murky lighting.

---

## 2. Final Project Folder Structure

The workspace is organized into a clean and structured layout to keep raw data, enhanced datasets, notebooks, and configurations completely isolated.

```text
underwater_drone_custom_model/
│
├── dataset/                  <-- Processed dataset (Train/Val + YOLO Labels)
│   ├── annotations/          <-- Original COCO JSON files
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
│
├── dataset_enhanced/         <-- Preprocessed dataset ready for YOLO training
│   ├── images/
│   │   ├── train/            <-- Fully enhanced with the Refined Moody Spotlight Pipeline
│   │   └── val/              <-- Fully enhanced with the Refined Moody Spotlight Pipeline
│   └── labels/               <-- Copied YOLO label text files (.txt)
│
├── notebooks/                <-- Jupyter Notebooks directory
│   ├── underwater_preprocessing.ipynb
│   └── train_baseline.ipynb
│
├── original_data/            <-- Raw source dataset archive files
│   ├── annotations/
│   └── images/
│
├── dataset.yaml              <-- YOLO configuration file mapping paths, classes, and names
├── prepare_data.py           <-- Automated data ingestion, COCO-to-YOLO conversion, and bounding-box script
├── requirements.txt          <-- Project dependency definitions
├── .gitignore                <-- Git exclusion rules for large datasets and model weights
└── README.md                 <-- Project documentation
```

---

## 3. Data Preparation & Ingestion Methods (`prepare_data.py`)

Before image enhancement, raw data undergoes an automated processing pipeline to structure and convert annotations into standard YOLO format.

- **Dataset Discovery & Verification** — Automatically scanned and searched directory paths to locate COCO-format JSON annotation files and raw image directories.
- **COCO to YOLO Label Conversion** — Parsed JSON bounding boxes (`[x_min, y_min, w, h]`) and converted them into normalized YOLO format (`[class_id, x_center, y_center, width, height]`) relative to image dimensions.
- **Coordinate Bounds Checking** — Applied strict limiting functions (`min(max(val, 0.0), 1.0)`) to prevent out-of-bounds normalized bounding-box coordinates.
- **Directory Structuring** — Programmatically created and organized standard YOLOv8 folder hierarchies (`dataset/images/train`, `dataset/images/val`, `dataset/labels/train`, and `dataset/labels/val`).
- **Automated Dataset Configuration (`dataset.yaml`)** — Extracted category names dynamically from the JSON metadata and auto-generated the required YAML configuration file containing path mappings and class counts.

---

## 4. Preprocessing & Enhancement Methods Used in "YOLO Custom Model"

To optimize visual inputs for the **YOLO Custom Model**, images undergo the **Refined Moody Spotlight Pipeline**, which combines six distinct computer vision techniques implemented using OpenCV and NumPy.

1. **Grayworld Color Balance** — Computes the mean intensities of the Blue, Green, and Red channels and scales them uniformly to eliminate the dominant blue-green chromatic cast typical of underwater imagery.
2. **LAB Color Space CLAHE** — Converts the color-balanced frame into the LAB color space and applies Contrast Limited Adaptive Histogram Equalization (CLAHE) specifically to the luminance (L) channel, boosting local target contrast without amplifying background specular noise.
3. **Balanced Gamma Correction** — Applies a precise gamma transformation (γ = 0.55) through a lookup table (LUT) to suppress midtone haze and create a deep, moody spotlight effect where the background falls into shadow while targets stand out.
4. **Controlled Red Channel Boost** — Mildly increases intensity values in the red channel (ΔR = +8) to restore natural biological color warmth lost due to underwater light wavelength attenuation.
5. **Advanced Denoising** — Filters out high-frequency granular noise and floating particulate "snow" using Non-Local Means Denoising (`fastNlMeansDenoisingColored`) combined with a **3 × 3 median blur**.
6. **Crisp Unsharp Masking** — Combines a Gaussian blur filter with weighted image addition (`cv2.addWeighted`) to sharpen target boundaries and contours for optimal neural network feature extraction.

---

## 5. Model Comparison: Normal YOLOv8n Base Model vs. YOLO Custom Model

When it comes to input data, the Normal YOLOv8n Base Model works with raw, unprocessed underwater images that carry severe color casts, while the YOLO Custom Model works with professionally enhanced images produced through the Refined Moody Spotlight Pipeline.

In terms of handling backscatter, the base model struggles with floating particles ("snow") and murky fog in the water, whereas the Custom Model actively filters out this background snow using NLM (Non-Local Means) denoising combined with deep gamma shadows.

On object contrast, the base model suffers from low local contrast, which causes small or camouflaged objects to blend into the background. The Custom Model, by contrast, achieves high local contrast through LAB CLAHE and unsharp masking, making objects stand out sharply from their surroundings.

Finally, in terms of detection performance, the base model shows higher false negative and false positive rates due to poor visibility and obscured features. The Custom Model, on the other hand, delivers optimized bounding box localization and a higher mean Average Precision (mAP) on underwater targets.
---

## 6. Model Comparison: "YOLO Dark Water" vs. "YOLO Custom Model"

The **YOLO Dark Water** model focused entirely on robust data engineering, including automated dataset discovery, label parsing, bounding-box normalization, coordinate bounds checking, and directory structure generation. Its core methodology relied on automated **Dataset Discovery & Verification**, accurate **COCO-to-YOLO Label Conversion**, safe **Coordinate Bounds Checking**, and standardized **Directory Structuring & YAML Generation** for building a reliable YOLO training dataset.

The **YOLO Custom Model** extends everything achieved in the **YOLO Dark Water** data ingestion framework and builds a sophisticated visual enhancement layer on top of it. While **YOLO Dark Water** perfected structural preparation and coordinate integrity, the **YOLO Custom Model** introduces an advanced pixel-level preprocessing pipeline consisting of **Grayworld Color Balance**, **LAB CLAHE**, **Gamma Spotlight Shaping**, **Controlled Red Channel Boost**, **Non-Local Means Denoising**, and **Unsharp Masking** to systematically transform underwater image clarity before training.