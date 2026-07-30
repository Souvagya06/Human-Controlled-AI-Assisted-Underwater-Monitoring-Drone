Human-Controlled AI-Assisted Underwater Monitoring Drone: YOLO Custom Model
The YOLO Custom Model is an advanced object detection iteration built on top of the YOLOv8 architecture, specifically designed to tackle the severe visibility degradation and color attenuation found in deep-sea environments.

Unlike standard models trained on generic imagery or simple contrast adjustments, this model integrates a meticulously crafted, multi-step underwater computer vision preprocessing pipeline combined with automated COCO-to-YOLO conversion and coordinate normalization. Furthermore, it incorporates state-of-the-art architectural upgrades (Tier 1–3 enhancements) including a P2 small-object detection head, CBAM (Convolutional Block Attention Module), BiFPN weighted feature fusion, Wise-IoU v3 loss, and Copy-Paste augmentation. This ensures that the network learns to detect objects such as marine species and underwater debris accurately even when surrounded by heavy backscatter, chromatic casts, and murky lighting.

2. Final Project Folder Structure
The workspace is organized into a clean and structured layout to keep raw data, enhanced datasets, custom configurations, source modules, and notebooks completely isolated.

Plaintext
underwater_drone_custom_model/
├── dataset/                  
│   ├── annotations/          
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
├── dataset_enhanced/         
│   ├── images/
│   │   ├── train/            
│   │   └── val/
│   └── labels/               
├── notebooks/                
│   ├── 01_underwater_preprocessing.ipynb
│   ├── 02_train_baseline.ipynb
│   └── 03_train_custom_v1.ipynb  
├── original_data/            
│   ├── annotations/
│   └── images/
├── attention.py              
├── custom_loss.py            
├── yolo-custom.yaml          
├── dataset.yaml              
├── dataset_enhanced.yaml     
├── prepare_data.py           
├── requirements.txt          
├── .gitignore                
└── README.md                 
The structured layout ensures that all raw data files, preprocessed outputs, architecture definitions, and custom training scripts remain cleanly separated and easily maintainable across different development branches.

3. Data Preparation & Ingestion Methods (prepare_data.py)
Before image enhancement, raw data undergoes an automated processing pipeline to structure and convert annotations into standard YOLO format:

Dataset Discovery & Verification — Automatically scanned and searched directory paths to locate COCO-format JSON annotation files and raw image directories.

COCO to YOLO Label Conversion — Parsed JSON bounding boxes ([x_min, y_min, w, h]) and converted them into normalized YOLO format ([class_id, x_center, y_center, width, height]) relative to image dimensions.

Coordinate Bounds Checking — Applied strict limiting functions (min(max(val, 0.0), 1.0)) to prevent out-of-bounds normalized bounding-box coordinates.

Directory Structuring — Programmatically created and organized standard YOLOv8 folder hierarchies (dataset/images/train, dataset/images/val, dataset/labels/train, and dataset/labels/val).

Automated Dataset Configuration (dataset.yaml) — Extracted category names dynamically from the JSON metadata and auto-generated the required YAML configuration file containing path mappings and class counts.

4. Preprocessing & Enhancement Methods Used in "YOLO Custom Model"
To optimize visual inputs for the YOLO Custom Model, images undergo the Refined Moody Spotlight Pipeline, which combines six distinct computer vision techniques implemented using OpenCV and NumPy:

Grayworld Color Balance — Computes the mean intensities of the Blue, Green, and Red channels and scales them uniformly to eliminate the dominant blue-green chromatic cast typical of underwater imagery.

LAB Color Space CLAHE — Converts the color-balanced frame into the LAB color space and applies Contrast Limited Adaptive Histogram Equalization (CLAHE) specifically to the luminance channel, boosting local target contrast without amplifying background specular noise.

Balanced Gamma Correction — Applies a precise gamma transformation (γ=0.55) through a lookup table (LUT) to suppress midtone haze and create a deep, moody spotlight effect where the background falls into shadow while targets stand out.

Controlled Red Channel Boost — Mildly increases intensity values in the red channel (ΔR=+8) to restore natural biological color warmth lost due to underwater light wavelength attenuation.

Advanced Denoising — Filters out high-frequency granular noise and floating particulate "snow" using Non-Local Means Denoising (fastNlMeansDenoisingColored) combined with a 3×3 median blur.

Crisp Unsharp Masking — Combines a Gaussian blur filter with weighted image addition (cv2.addWeighted) to sharpen target boundaries and contours for optimal neural network feature extraction.

5. Advanced Architecture & Training Injections (Tiers 1–3)
To maximize performance on tiny targets, severe class imbalances, and complex underwater backgrounds, the following enhancements have been injected directly into the custom pipeline:

Tier 1 (Core Foundations): CBAM attention injected at the P3 scale, Wise-IoU v3 bounding box regression, a dedicated P2 small-object detection head, Cosine Learning Rate scheduling, EMA, AMP, and disabled/restricted heavy HSV color augmentations.

Tier 2 (Advanced Feature Fusion & Loss): BiFPN weighted feature fusion in the neck to replace plain concatenation, and multi-scale training support for enhanced scale-invariance.

Tier 3 (Robust Augmentations): Copy-Paste augmentation to heavily diversify rare marine debris and class instances.

6. Model Comparison: Normal YOLOv8n Base Model vs. YOLO Custom Model
Input Data: The Normal YOLOv8n Base Model works with raw, unprocessed underwater images that carry severe color casts, whereas the YOLO Custom Model works with professionally enhanced images produced through the Refined Moody Spotlight Pipeline.

Handling Backscatter: The base model struggles with floating particles ("snow") and murky fog in the water, whereas the Custom Model actively filters out background snow using NLM denoising combined with deep gamma shadows.

Object Contrast: The base model suffers from low local contrast, causing small or camouflaged objects to blend into the background. The Custom Model achieves high local contrast through LAB CLAHE and unsharp masking, making objects stand out sharply.

Detection Performance: The base model shows higher false negative and false positive rates due to poor visibility and obscured features. The Custom Model delivers optimized bounding box localization, enhanced small-object detection via the P2 head, and a higher mean Average Precision (mAP) on underwater targets.

7. Model Comparison: "YOLO Dark Water" vs. "YOLO Custom Model"
YOLO Dark Water: Focused entirely on robust data engineering, including automated dataset discovery, label parsing, bounding-box normalization, coordinate bounds checking, and directory structure generation. Its core methodology relied on structural preparation, coordinate integrity, and standardized YAML generation for building a reliable training baseline dataset.

YOLO Custom Model: Extends everything achieved in the data ingestion framework and builds a sophisticated visual enhancement layer combined with advanced network architecture modifications on top of it. While YOLO Dark Water perfected structural preparation, the YOLO Custom Model introduces both the pixel-level Refined Moody Spotlight Pipeline and state-of-the-art Tier 1–3 architectural upgrades (BiFPN, CBAM, P2 head, and Wise-IoU) to systematically maximize detection accuracy in real-world underwater environments.